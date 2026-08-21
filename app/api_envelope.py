"""Error-envelope selection shared by the auth/rate-limit/model-access paths.

429 and 403 bodies must match the shape of the API surface the request arrived
on, so clients' SDKs can parse them. This lived as three byte-identical copies
(``app/auth/middleware.py``, ``app/rate_limit_dep.py``, ``app/model_access_dep.py``);
keeping one copy here removes the risk of them drifting apart.

Deliberately imports nothing from ``app.*`` — the three call sites already form
an import chain (``app.auth.middleware`` -> ``app.rate_limit``,
``app.rate_limit_dep`` -> ``app.auth.admin``) that a shared dependency would
otherwise turn into a cycle.
"""

from typing import Optional


def envelope_for(path: str, override: Optional[str]) -> str:
    """Pick the error-body shape ("anthropic" / "azure" / "openai") for *path*.

    *path* must be the app-relative path — the same one the sub-app sees on its
    dedicated port. ``MountedApp`` guarantees that for mounted requests.
    """
    if override:
        return override
    if path.startswith("/v1/messages"):
        return "anthropic"
    if path.startswith("/openai/"):
        return "azure"
    return "openai"
