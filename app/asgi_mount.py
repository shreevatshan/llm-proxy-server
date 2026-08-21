"""ASGI helpers for serving the provider apps under the management port.

``create_management_app`` mounts the three provider apps (``/openai``,
``/anthropic``, ``/azure-openai``) so every API is reachable through the single
management port as well as its own dedicated port. Two things have to be fixed
up for a mounted request to behave *identically* to the same request on the
dedicated port; this module holds both.

1. ``MountedApp`` — Starlette's ``Mount`` does **not** strip the prefix from
   ``scope["path"]`` (it only sets ``root_path``), and ``URL(scope=...)`` is
   built from ``scope["path"]`` alone. So ``request.url.path`` inside a mounted
   sub-app is the *prefixed* path, and every path-based decision the sub-app
   makes silently diverges — request tracking, usage counting, model-alias
   rewriting, rate-limit skip lists and error-envelope selection all key off it.
   ``MountedApp`` normalizes the scope so the sub-app sees exactly what it sees
   standalone.

2. ``CORSExceptPrefixes`` — the management app needs strict, credentialed CORS
   for its own cookie-authenticated routes, but the mounted provider APIs must
   keep the permissive non-credentialed policy they serve on their own ports.
"""

from urllib.parse import urlsplit, urlunsplit

from starlette.datastructures import MutableHeaders

# Path prefixes the provider apps are mounted under on the management port.
# Order matches (openai_app, anthropic_app, azure_openai_app).
MOUNTED_API_PREFIXES = ("/openai", "/anthropic", "/azure-openai")


def _under(path: str, prefix: str) -> bool:
    """Whether *path* is *prefix* itself or lies beneath it.

    Segment-aware on purpose: a plain ``startswith`` would treat
    ``/openai-evil/x`` as living under ``/openai``.
    """
    return path == prefix or path.startswith(prefix + "/")


class MountedApp:
    """Present *app* under *prefix* while handing it its standalone scope.

    Wrap a sub-app with this before mounting it so that ``request.url.path``,
    and everything downstream of it, matches what the sub-app would see when
    served directly on its own port.
    """

    def __init__(self, app, prefix: str):
        self.app = app
        self.prefix = prefix.rstrip("/")

    @property
    def routes(self):
        """Expose the sub-app's routes so ``url_path_for`` can still recurse.

        Without this, wrapping the app in an opaque object breaks
        ``request.url_for(...)`` with ``NoMatchFound``, because the parent's
        ``Mount`` can no longer walk into the child's router.
        """
        return getattr(self.app, "routes", [])

    def _reprefix_location(
        self, location: str, status: int, host: str, parent_root: str = ""
    ) -> str:
        """Re-add the mount prefix to a redirect the sub-app generated.

        The sub-app builds redirects from the (now stripped) path, so they come
        back out missing the prefix. Relative redirects are always prefixed.
        Absolute ones only when they point back at us *and* carry Starlette's
        own ``redirect_slashes`` status — the provider apps deliberately issue
        302s to the management origin, and those must pass through untouched.

        *parent_root* is the ASGI ``root_path`` the request arrived with, before
        the mount appended our prefix. Redirects are built from ``scope["path"]``,
        which carries that root, so the prefix goes back in *after* it rather
        than at the front of the path.
        """
        parts = urlsplit(location)
        if not parts.path.startswith("/"):
            return location
        mount_path = parent_root + self.prefix
        if _under(parts.path, mount_path):
            return location  # already prefixed
        if parent_root and not _under(parts.path, parent_root):
            return location  # not a path this mount produced
        if parts.netloc and (status not in (307, 308) or parts.netloc != host):
            return location
        new_path = mount_path + parts.path[len(parent_root):]
        return urlunsplit(
            (parts.scheme, parts.netloc, new_path, parts.query, parts.fragment)
        )

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        host = next(
            (v.decode("latin-1") for k, v in scope.get("headers", []) if k == b"host"),
            "",
        )

        # Copy: the parent's scope must not be mutated.
        scope = dict(scope)
        path = scope.get("path", "")
        root_path = scope.get("root_path", "")

        # The ASGI path carries root_path, and Mount appends our prefix to
        # root_path without touching the path — so behind a reverse-proxy
        # --root-path the prefix sits *after* the parent's root, not at the
        # front. Locate it there, and hand the sub-app back the root_path the
        # parent had before mounting so --root-path still reaches it intact.
        parent_root = (
            root_path[: len(root_path) - len(self.prefix)]
            if root_path.endswith(self.prefix)
            else root_path
        )
        mount_path = parent_root + self.prefix

        # Guarded so this becomes a no-op if a future Starlette strips the path
        # itself, rather than eating a second prefix. Path and root_path move
        # together: unwinding one without the other leaves the sub-app matching
        # routes against a path that still holds the prefix, i.e. a blanket 404.
        if _under(path, mount_path):
            scope["path"] = parent_root + (path[len(mount_path):] or "/")
            raw_mount = mount_path.encode("latin-1")
            raw_path = scope.get("raw_path")
            if raw_path and raw_path.startswith(raw_mount):
                scope["raw_path"] = (
                    parent_root.encode("latin-1") + (raw_path[len(raw_mount):] or b"/")
                )
            scope["root_path"] = parent_root

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                location = headers.get("location")
                if location:
                    headers["location"] = self._reprefix_location(
                        location, message["status"], host, parent_root
                    )
            await send(message)

        await self.app(scope, receive, send_wrapper)


class CORSExceptPrefixes:
    """Apply CORS to every request except those under *exempt_prefixes*.

    Requests to the mounted provider APIs bypass the management app's strict
    credentialed CORS and are handled by the sub-app's own permissive policy,
    matching what those APIs serve on their dedicated ports.

    *cors_app_factory* is called with the wrapped app and must return the
    configured CORS middleware; a factory (rather than a ready instance) is
    needed because ``add_middleware`` only supplies the inner app at build time.
    """

    def __init__(self, app, cors_app_factory, exempt_prefixes):
        self.app = app
        self.cors_app = cors_app_factory(app)
        self.exempt_prefixes = tuple(p.rstrip("/") for p in exempt_prefixes)

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            path = scope.get("path", "")
            if any(_under(path, prefix) for prefix in self.exempt_prefixes):
                await self.app(scope, receive, send)
                return
        await self.cors_app(scope, receive, send)
