"""
Codex auth helper tests.

Covers:
  - _decode_jwt_payload extracts claims.
  - _seconds_until_expiry math.
  - CODEX_HOME env var override resolves auth.json path.
  - _lock_file_path is the sibling lockfile of auth.json.
  - get_bearer fast path (token fresh) does NOT call refresh.
  - get_bearer slow path acquires the interprocess lock.
  - get_bearer double-checked locking: if auth.json was refreshed while we
    waited for the lock, skip our own refresh.
  - get_bearer force_refresh=True bypasses the freshness check.
  - get_account_id / get_chatgpt_plan read JWT claims.

All tests use an isolated CODEX_HOME (tmp_path) so they never touch the
user's real ~/.codex/auth.json.
"""
import base64
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

src = Path(__file__).resolve().parent.parent
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from IkaModel.codex import auth as codex_auth


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _make_jwt(payload: dict) -> str:
    """Build a fake JWT (header.payload.signature) — signature is bogus, but
    we never validate it; the codex backend would, but our local code only
    decodes the payload to read exp / chatgpt_plan_type."""
    def _b64(d):
        return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()
    return f"{_b64({'alg':'RS256','typ':'JWT'})}.{_b64(payload)}.fakesig"


def _write_auth_json(home: Path, *, exp_offset_sec: int = 3600,
                     plan: str = "pro", account_id: str = "acc-xyz",
                     access_token: str = None, refresh_token: str = "rt-stored"):
    home.mkdir(parents=True, exist_ok=True)
    if access_token is None:
        access_token = _make_jwt({
            "exp": int(time.time() + exp_offset_sec),
            "aud": ["https://api.openai.com/v1"],
            "client_id": "app_test",
            "https://api.openai.com/auth": {
                "chatgpt_plan_type": plan,
                "chatgpt_account_id": account_id,
            },
        })
    auth_data = {
        "auth_mode": "chatgpt",
        "OPENAI_API_KEY": None,
        "tokens": {
            "id_token": _make_jwt({"sub": "u"}),
            "access_token": access_token,
            "refresh_token": refresh_token,
            "account_id": account_id,
        },
        "last_refresh": "2026-01-01T00:00:00Z",
    }
    (home / "auth.json").write_text(json.dumps(auth_data, indent=2))
    return auth_data


@pytest.fixture
def isolated_codex_home(tmp_path, monkeypatch):
    """Point CODEX_HOME at a temp dir so we never touch ~/.codex."""
    home = tmp_path / "codex_home"
    monkeypatch.setenv("CODEX_HOME", str(home))
    return home


# ----------------------------------------------------------------------
# JWT decode + expiry
# ----------------------------------------------------------------------

class TestJWTHelpers:

    def test_decode_jwt_payload(self):
        tok = _make_jwt({"exp": 12345, "foo": "bar"})
        assert codex_auth._decode_jwt_payload(tok) == {"exp": 12345, "foo": "bar"}

    def test_decode_jwt_payload_invalid_returns_empty(self):
        assert codex_auth._decode_jwt_payload("not-a-jwt") == {}

    def test_seconds_until_expiry(self):
        tok = _make_jwt({"exp": int(time.time() + 100)})
        v = codex_auth._seconds_until_expiry(tok)
        assert 95 <= v <= 105

    def test_seconds_until_expiry_missing_exp(self):
        tok = _make_jwt({})
        assert codex_auth._seconds_until_expiry(tok) == 0.0


# ----------------------------------------------------------------------
# Auth path resolution + lockfile location
# ----------------------------------------------------------------------

class TestPaths:

    def test_codex_home_env_var_respected(self, isolated_codex_home):
        assert codex_auth._auth_json_path() == isolated_codex_home / "auth.json"

    def test_lock_file_path_is_sibling(self, isolated_codex_home):
        lock = codex_auth._lock_file_path()
        assert lock == isolated_codex_home / "auth.json.lock"
        assert lock.parent == codex_auth._auth_json_path().parent


# ----------------------------------------------------------------------
# get_bearer happy paths
# ----------------------------------------------------------------------

class TestGetBearer:

    def test_fresh_token_no_refresh(self, isolated_codex_home):
        _write_auth_json(isolated_codex_home, exp_offset_sec=24 * 3600)
        with patch.object(codex_auth, "_refresh_and_persist") as m:
            bearer = codex_auth.get_bearer()
        m.assert_not_called()
        assert bearer.startswith("ey")  # base64-encoded JWT header

    def test_near_expiry_triggers_refresh(self, isolated_codex_home):
        _write_auth_json(isolated_codex_home, exp_offset_sec=60)  # 1 min — < 8-min lead
        refreshed = {
            "auth_mode": "chatgpt",
            "tokens": {
                "access_token": _make_jwt({"exp": int(time.time() + 24 * 3600)}),
                "refresh_token": "rt-new",
                "id_token": _make_jwt({}),
                "account_id": "acc",
            },
            "last_refresh": "2099-01-01T00:00:00Z",
        }
        with patch.object(codex_auth, "_refresh_and_persist", return_value=refreshed) as m:
            bearer = codex_auth.get_bearer()
        m.assert_called_once()
        # Should return the NEW token from the refresh result
        assert bearer == refreshed["tokens"]["access_token"]

    def test_force_refresh_bypasses_freshness_check(self, isolated_codex_home):
        _write_auth_json(isolated_codex_home, exp_offset_sec=24 * 3600)  # fresh
        refreshed = {
            "auth_mode": "chatgpt",
            "tokens": {
                "access_token": _make_jwt({"exp": int(time.time() + 24 * 3600)}),
                "refresh_token": "rt-new",
                "id_token": _make_jwt({}),
                "account_id": "acc",
            },
            "last_refresh": "2099-01-01T00:00:00Z",
        }
        with patch.object(codex_auth, "_refresh_and_persist", return_value=refreshed) as m:
            codex_auth.get_bearer(force_refresh=True)
        m.assert_called_once()

    def test_double_checked_locking_skips_redundant_refresh(self, isolated_codex_home):
        """Simulate: another process refreshed while we waited for the file
        lock. After we acquire the lock and re-read, the token is fresh, so
        we must NOT call refresh ourselves."""
        # On disk: still-expired token (what we read pre-lock)
        _write_auth_json(isolated_codex_home, exp_offset_sec=60)
        read_calls = {"n": 0}
        real_seconds = codex_auth._seconds_until_expiry

        def fake_seconds(tok):
            # Pre-lock check: expired (force slow path).
            # Post-lock re-read: fresh (simulate other process refreshed).
            read_calls["n"] += 1
            if read_calls["n"] == 1:
                return 0
            return 24 * 3600

        with patch.object(codex_auth, "_seconds_until_expiry", side_effect=fake_seconds), \
             patch.object(codex_auth, "_refresh_and_persist") as m:
            codex_auth.get_bearer()
        m.assert_not_called()
        assert read_calls["n"] == 2  # one pre-lock, one post-lock

    def test_apikey_mode_returns_key_no_refresh(self, isolated_codex_home):
        isolated_codex_home.mkdir(parents=True, exist_ok=True)
        (isolated_codex_home / "auth.json").write_text(json.dumps({
            "auth_mode": "apikey",
            "OPENAI_API_KEY": "sk-test-12345",
            "tokens": None, "last_refresh": "",
        }))
        with patch.object(codex_auth, "_refresh_and_persist") as m:
            bearer = codex_auth.get_bearer()
        m.assert_not_called()
        assert bearer == "sk-test-12345"

    def test_missing_auth_json_raises(self, isolated_codex_home):
        """auth.json absent → clean error pointing user at `codex login`."""
        with pytest.raises(FileNotFoundError, match="codex login"):
            codex_auth.get_bearer()


# ----------------------------------------------------------------------
# Account / plan claim extraction
# ----------------------------------------------------------------------

class TestClaims:

    def test_get_account_id(self, isolated_codex_home):
        _write_auth_json(isolated_codex_home, account_id="acc-abc123")
        assert codex_auth.get_account_id() == "acc-abc123"

    def test_get_chatgpt_plan(self, isolated_codex_home):
        _write_auth_json(isolated_codex_home, plan="pro")
        assert codex_auth.get_chatgpt_plan() == "pro"

    def test_get_account_id_missing_auth_returns_none(self, isolated_codex_home):
        # No auth.json written.
        assert codex_auth.get_account_id() is None

    def test_get_chatgpt_plan_missing_auth_returns_none(self, isolated_codex_home):
        assert codex_auth.get_chatgpt_plan() is None
