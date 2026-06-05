"""
Optional helper for callers that want to source the codex bearer from the
Codex CLI's stored credentials (``~/.codex/auth.json``) with automatic
OAuth refresh.

This module is *not* used by the codex provider request path itself — the
provider treats ``BareBoneModel.api_key`` as a literal bearer, matching every
other IkaCore provider. Use ``get_bearer()`` here only if you want the
convenience of "read the token the Codex CLI already minted, refresh it if
it's about to expire":

    from IkaModel.codex import codex_auth
    bearer = codex_auth.get_bearer()              # may refresh
    agent  = IkaBaseAgent(..., api_key=bearer, api_url=CODEX_API_URL)

Or following the env-var convention used by other providers:

    bearer = os.getenv("CODEX_BEARER") or codex_auth.get_bearer()

Only the refresh path talks to ``auth.openai.com`` and uses the public Codex
OAuth client id; the codex Responses endpoint itself only needs the bearer.

Refresh wire format (mirrors ``codex-rs/login``):

    POST https://auth.openai.com/oauth/token
        grant_type=refresh_token
        refresh_token=<stored>
        client_id=app_EMoamEEZ73f0CkXaXp7hrann   (public Codex OAuth client)
        scope=openid profile email offline_access

Refreshed tokens are persisted back to ``auth.json``. ``get_bearer()`` uses a
module-level ``threading.Lock`` for same-process callers and an exclusive
``fcntl.flock`` on a sibling lockfile for IkaCore processes on Unix/macOS. It
re-reads ``auth.json`` after acquiring the file lock so a second process can
adopt an already-refreshed token instead of using a stale refresh token. The
Codex CLI does not use this lockfile, so concurrent CLI refreshes cannot be
fully coordinated here.
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import httpx

LOG = logging.getLogger(__name__)

# Public Codex OAuth client id (no client_secret — Codex is a public/native
# OAuth client). Visible in any codex-issued JWT's ``client_id`` claim and
# declared in codex-rs/login/src/auth/manager.rs.
CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
_CODEX_TOKEN_URL_OVERRIDE_ENV = "CODEX_REFRESH_TOKEN_URL_OVERRIDE"
_CODEX_TOKEN_URL_OVERRIDE_ALLOW_ENV = "IKACORE_ALLOW_CODEX_TOKEN_URL_OVERRIDE"
CODEX_AUTH_SCOPE = "openid profile email offline_access"

# Refresh when the access_token is within this many seconds of its ``exp``.
# Codex itself uses 8 minutes; we mirror it.
REFRESH_LEAD_SECONDS = 8 * 60

_lock = threading.Lock()
_fcntl_warned = False


def _resolve_oauth_token_url() -> str:
    override = os.environ.get(_CODEX_TOKEN_URL_OVERRIDE_ENV)
    if not override:
        return CODEX_OAUTH_TOKEN_URL
    if os.environ.get(_CODEX_TOKEN_URL_OVERRIDE_ALLOW_ENV) == "1":
        return override
    LOG.warning(
        "Ignoring %s because %s=1 is not set; using the official Codex OAuth token URL.",
        _CODEX_TOKEN_URL_OVERRIDE_ENV,
        _CODEX_TOKEN_URL_OVERRIDE_ALLOW_ENV,
    )
    return CODEX_OAUTH_TOKEN_URL


def _lock_file_path() -> Path:
    """Sibling lockfile used to serialize refresh across IkaCore processes."""
    p = _auth_json_path()
    return p.with_name(p.name + ".lock")


@contextlib.contextmanager
def _interprocess_lock() -> Iterator[None]:
    """
    Acquire an exclusive interprocess lock for the duration of a refresh.

    Used to prevent two IkaCore processes from racing to refresh the same
    Codex bearer (the OAuth server may rotate the refresh_token, in which
    case the loser's stored refresh_token is invalidated and the auth.json
    can end up in a broken state). Codex CLI itself does not use file
    locking on auth.json, so we cannot fully coordinate with the Rust
    process — this protects IkaCore-vs-IkaCore concurrency only.

    Implementation: ``fcntl.flock`` on a sibling lockfile (Unix/macOS).
    On Windows ``fcntl`` is unavailable; we fall back to the in-process
    lock alone and log a one-time warning.
    """
    global _fcntl_warned
    lock_path = _lock_file_path()
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    try:
        import fcntl  # type: ignore[import-not-found]
    except ImportError:
        if not _fcntl_warned:
            LOG.warning(
                "fcntl unavailable on this platform; codex auth.json refresh "
                "has only in-process locking. Concurrent IkaCore processes "
                "may race when the bearer expires."
            )
            _fcntl_warned = True
        yield
        return

    fd = None
    try:
        # 'a+' so the file is created if missing without truncating its
        # contents (the file itself is just a lock anchor; we never read it).
        fd = open(lock_path, "a+")
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if fd is not None:
            try:
                fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                fd.close()
            except Exception:
                pass


def _auth_json_path() -> Path:
    home = os.environ.get("CODEX_HOME") or os.path.expanduser("~/.codex")
    return Path(home) / "auth.json"


def _read_auth() -> Dict[str, Any]:
    path = _auth_json_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Codex auth file not found at {path}. "
            "Run `codex login` (or set CODEX_HOME) before using the codex provider."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_auth(data: Dict[str, Any]) -> None:
    path = _auth_json_path()
    # Atomic write so a crash mid-write doesn't trash the user's auth.
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def _decode_jwt_payload(token: str) -> Dict[str, Any]:
    # Header.Payload.Signature — we only need the payload, no signature check
    # (the server validates; we just want ``exp`` so we know when to refresh).
    try:
        payload_segment = token.split(".")[1]
        padded = payload_segment + "=" * (-len(payload_segment) % 4)
        return json.loads(base64.urlsafe_b64decode(padded))
    except Exception as e:
        LOG.warning(f"Failed to decode JWT payload: {e}")
        return {}


def _seconds_until_expiry(token: str) -> float:
    payload = _decode_jwt_payload(token)
    exp = payload.get("exp")
    if not isinstance(exp, (int, float)):
        return 0.0
    return float(exp) - time.time()


def _refresh_tokens(refresh_token: str) -> Dict[str, Any]:
    """Exchange a refresh_token for a fresh access_token/id_token pair."""
    token_url = _resolve_oauth_token_url()
    LOG.info("Refreshing Codex bearer via %s", token_url)
    body = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": CODEX_OAUTH_CLIENT_ID,
        "scope": CODEX_AUTH_SCOPE,
    }
    resp = httpx.post(
        token_url,
        json=body,
        headers={"Content-Type": "application/json"},
        timeout=30.0,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Codex refresh failed: {resp.status_code} {resp.text[:500]}"
        )
    return resp.json()


def _refresh_and_persist() -> Dict[str, Any]:
    """Pull current tokens, refresh, write back. Returns the new auth dict."""
    data = _read_auth()
    tokens = data.get("tokens") or {}
    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        raise RuntimeError(
            "Codex auth.json has no refresh_token; cannot refresh access_token. "
            "Re-run `codex login`."
        )
    new = _refresh_tokens(refresh_token)
    # The OAuth response uses snake_case access_token, refresh_token, id_token.
    tokens["access_token"] = new.get("access_token", tokens.get("access_token"))
    tokens["id_token"] = new.get("id_token", tokens.get("id_token"))
    # The server may rotate the refresh_token; if it does, persist the new one.
    if new.get("refresh_token"):
        tokens["refresh_token"] = new["refresh_token"]
    data["tokens"] = tokens
    data["last_refresh"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_auth(data)
    return data


def get_bearer(force_refresh: bool = False) -> str:
    """
    Return a valid Codex bearer token, refreshing if it expires soon (or
    when ``force_refresh`` is true — used after a 401).

    Concurrency safety:
      - In-process: a module-level ``threading.Lock`` serializes refreshes
        across threads in the same process.
      - Cross-process: an exclusive ``fcntl.flock`` on a sibling lockfile
        serializes refreshes across IkaCore processes. We do not coordinate
        with the Codex CLI (it doesn't use file locking on auth.json), but
        any IkaCore vs IkaCore race is prevented.
      - Double-checked locking: after acquiring the interprocess lock we
        re-read auth.json so that if another process refreshed while we
        were waiting, we adopt the new token instead of refreshing again
        (important: most OAuth servers rotate refresh_token, so a second
        refresh with the now-stale refresh_token would fail and may leave
        auth.json in a bad state).
    """
    with _lock:
        data = _read_auth()
        mode = (data.get("auth_mode") or "").lower()
        tokens = data.get("tokens") or {}

        if mode == "apikey":
            # API-key mode: the literal key is the bearer; nothing to refresh.
            key = data.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("auth.json is in apikey mode but OPENAI_API_KEY is missing.")
            return key

        access = tokens.get("access_token")
        if not access:
            # Maybe the file's in apikey mode without auth_mode set explicitly.
            key = data.get("OPENAI_API_KEY")
            if key:
                return key
            raise RuntimeError("Codex auth.json has neither tokens.access_token nor OPENAI_API_KEY.")

        # Fast path: token still fresh and caller didn't force a refresh.
        if not force_refresh and _seconds_until_expiry(access) >= REFRESH_LEAD_SECONDS:
            return access

        # Slow path: refresh under an interprocess lock with a double-check.
        with _interprocess_lock():
            data = _read_auth()  # re-read after acquiring lock
            tokens = data.get("tokens") or {}
            access = tokens.get("access_token")
            # If another process refreshed while we were waiting, adopt theirs.
            if (
                not force_refresh
                and access
                and _seconds_until_expiry(access) >= REFRESH_LEAD_SECONDS
            ):
                return access
            data = _refresh_and_persist()
            return data["tokens"]["access_token"]


def get_account_id() -> Optional[str]:
    """ChatGPT account id, or None if in api-key mode."""
    try:
        data = _read_auth()
    except FileNotFoundError:
        return None
    tokens = data.get("tokens") or {}
    return tokens.get("account_id")


def get_chatgpt_plan() -> Optional[str]:
    """Return the ChatGPT plan label (e.g. 'pro', 'plus') for telemetry/logging."""
    try:
        data = _read_auth()
    except FileNotFoundError:
        return None
    tokens = data.get("tokens") or {}
    access = tokens.get("access_token")
    if not access:
        return None
    payload = _decode_jwt_payload(access)
    auth_claims = payload.get("https://api.openai.com/auth") or {}
    return auth_claims.get("chatgpt_plan_type")
