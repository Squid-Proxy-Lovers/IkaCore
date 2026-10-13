"""Avoid storing credential fields in newly created runtime snapshots."""

# pyright: strict

import json
import re
from typing import Any, cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_SECRET_FIELDS = frozenset({"api_key", "apikey", "authorization", "password", "secret", "token",
                            "access_token", "refresh_token", "client_secret", "secret_key",
                            "x_api_key", "x_goog_api_key", "aws_secret_access_key"})
_CREDENTIAL = re.compile(r"\b(?:sk-(?:ant-|or-v1-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")
_URL_SECRETS = _SECRET_FIELDS | {"code", "key"}


def _safe_text(value: str) -> str:
    value = _CREDENTIAL.sub("[REDACTED]", value)
    if value.startswith(("http://", "https://")):
        try:
            parts = urlsplit(value)
            pairs = parse_qsl(parts.query, keep_blank_values=True)
            if "@" not in parts.netloc and not any(key.lower().replace("-", "_") in _URL_SECRETS for key, _ in pairs):
                return value
            netloc = parts.netloc.rsplit("@", 1)[-1] if "@" in parts.netloc else parts.netloc
            query = [(key, "[REDACTED]" if key.lower().replace("-", "_") in _URL_SECRETS else item)
                     for key, item in pairs]
            return urlunsplit((parts.scheme, netloc, parts.path, urlencode(query), parts.fragment))
        except ValueError:
            return "[REDACTED URL]"
    return value


def sanitize(value: object) -> object:
    if isinstance(value, dict):
        mapped = cast(dict[object, object], value)
        return {key: "[REDACTED]" if str(key).lower().replace("-", "_") in _SECRET_FIELDS else sanitize(item)
                for key, item in mapped.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in cast(list[object] | tuple[object, ...], value)]
    return _safe_text(value) if isinstance(value, str) else value


def snapshot_json(value: object) -> str:
    return json.dumps(sanitize(value))


def sanitized_state(state: dict[str, Any]) -> dict[str, Any]:
    safe = cast(dict[str, Any], sanitize(state))
    if json.dumps(safe) != json.dumps(state):
        safe["_redacted_fields"] = True
    return safe
