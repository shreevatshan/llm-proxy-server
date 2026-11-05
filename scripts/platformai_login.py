#!/usr/bin/env python3
"""Get a Zoho OAuth refresh token for a PlatformAI provider.

Runs the same browser login as ``zllm login``: OAuth 2.0 PKCE against Zoho
accounts with the OAuth client you pass via ``--client-id``; that client's
redirect URI must be ``http://localhost:4455/oauth/callback``. Run it on a machine with a browser,
then paste the printed refresh token into the admin dashboard's PlatformAI
provider form.

    python scripts/platformai_login.py --client-id 1000.XXXX
    python scripts/platformai_login.py --client-id 1000.XXXX --no-browser    # over SSH: paste the redirected URL back

Only the refresh token is printed; the access token is discarded.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.server
import json
import secrets
import socket
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
import webbrowser

# Default for the platformai.zoho.in gateway; pass --accounts-url for another data centre
# (e.g. https://accounts.zoho.com for platformai-labs.zoho.com).
DEFAULT_ACCOUNTS_URL = "https://accounts.zoho.in"
SCOPE = "PlatformAI.organizations.all AaaServer.profile.READ"
PORT = 4455
CALLBACK_PATH = "/oauth/callback"
REDIRECT_URI = f"http://localhost:{PORT}{CALLBACK_PATH}"


def _pkce() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def _authorize_url(accounts: str, client_id: str, challenge: str, state: str) -> str:
    params = {
        "scope": SCOPE,
        "client_id": client_id,
        "access_type": "offline",
        "prompt": "consent",
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "code_challenge_method": "S256",
        "code_challenge": challenge,
        "state": state,
    }
    return f"{accounts}/oauth/v2/auth?{urllib.parse.urlencode(params)}"


def _parse_callback(url: str) -> dict:
    query = urllib.parse.urlparse(url.strip()).query
    return {k: v[0] for k, v in urllib.parse.parse_qs(query).items()}


def _wait_for_callback() -> dict:
    result: dict = {}
    done = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (stdlib API)
            if not self.path.startswith(CALLBACK_PATH):
                self.send_response(404)
                self.end_headers()
                return
            result.update(_parse_callback(self.path))
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"<p>PlatformAI login complete. You can close this tab and return to the terminal.</p>")
            done.set()

        def log_message(self, *args):
            pass

    try:
        server = http.server.HTTPServer(("127.0.0.1", PORT), Handler)
    except OSError as exc:
        sys.exit(
            f"Port {PORT} is busy ({exc}). The Zoho client only accepts {REDIRECT_URI}; "
            "stop whatever holds the port (zcode-ai, a running zllm login) or use --no-browser."
        )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        done.wait()
    except KeyboardInterrupt:
        sys.exit("Cancelled.")
    finally:
        server.shutdown()
    return result


def _exchange(accounts: str, client_id: str, client_secret: str | None, code: str, verifier: str) -> dict:
    data = {
        "client_id": client_id,
        "grant_type": "authorization_code",
        "redirect_uri": REDIRECT_URI,
        "code_verifier": verifier,
        "code": code,
    }
    if client_secret:
        data["client_secret"] = client_secret
    request = urllib.request.Request(
        f"{accounts}/oauth/v2/token",
        data=urllib.parse.urlencode(data).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        sys.exit(f"Token exchange failed (HTTP {exc.code}): {body[:400]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--client-id", required=True, help="Zoho OAuth client ID")
    parser.add_argument("--client-secret", help="Only for a confidential client")
    parser.add_argument("--accounts-url", default=DEFAULT_ACCOUNTS_URL,
                        help=f"Zoho accounts host (default {DEFAULT_ACCOUNTS_URL})")
    parser.add_argument("--no-browser", action="store_true",
                        help="Print the login URL and read the redirected address from stdin")
    args = parser.parse_args()

    accounts = args.accounts_url.rstrip("/")
    client_id = args.client_id.strip()
    verifier, challenge = _pkce()
    state = secrets.token_hex(16)
    url = _authorize_url(accounts, client_id, challenge, state)

    if args.no_browser:
        print("Open this URL in a browser and approve:\n")
        print(url)
        print("\nThe browser then fails to load localhost:4455 — that is expected.")
        callback = _parse_callback(input("Paste the full address from the browser's address bar: "))
    else:
        # Check the port before sending the user off to log in.
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", PORT)) == 0:
                sys.exit(f"Port {PORT} is busy; stop zcode-ai / zllm login, or rerun with --no-browser.")
        print("Opening the Zoho login in your browser…")
        if not webbrowser.open(url):
            print(f"Could not open a browser; visit:\n{url}")
        callback = _wait_for_callback()

    if callback.get("error"):
        sys.exit(f"Login refused: {callback.get('error')} {callback.get('error_description', '')}".strip())
    if callback.get("state") != state:
        sys.exit("The callback does not belong to this login attempt (state mismatch). Run the script again.")
    if not callback.get("code"):
        sys.exit("The callback carried no authorization code.")

    tokens = _exchange(accounts, client_id, args.client_secret, callback["code"], verifier)
    refresh = tokens.get("refresh_token")
    if not refresh:
        sys.exit(f"Zoho returned no refresh token: {tokens.get('error') or 'unknown error'}")

    print("\nRefresh token (paste into the PlatformAI provider form; keep it secret):\n")
    print(refresh)
    print(f"\nAccounts: {accounts}   Client ID: {client_id}")


if __name__ == "__main__":
    main()
