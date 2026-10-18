"""Opt-in request/response dump files for chat diagnostics."""

# pyright: strict

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Optional

from IkaCore.agent_runtime_payloads import JsonDict

from ..base import BareBoneModel
from ..runtime_errors import IkaDebugDumpError

LOG = logging.getLogger(__name__)


def dump_api_round(
    barebone_model: BareBoneModel,
    payload: JsonDict,
    response_data: Optional[JsonDict],
    error: Optional[str] = None,
) -> None:
    out_dir = os.environ.get("IKA_DUMP_REQUESTS")
    if not out_dir:
        return
    try:
        os.makedirs(out_dir, exist_ok=True)
        agent = getattr(barebone_model, "agent_name", "agent") or "agent"
        agent_safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(agent))[:60]
        # Hash the conversation portion so duplicate prompts collide on disk.
        # Different providers use different field names: chat completions uses
        # "messages", the OpenAI Responses API uses "input", Anthropic uses
        # "messages" with its own shape. Hash whichever is present.
        convo: object = payload.get("messages") or payload.get("input") or []
        msg_repr = json.dumps(convo, sort_keys=True, default=str)
        sig = hashlib.sha256(msg_repr.encode("utf-8", "replace")).hexdigest()[:12]
        ts = f"{time.time():.3f}"
        fname = f"{ts}__{agent_safe}__{sig}.json"
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "agent_name": agent,
                    "model_id": getattr(barebone_model, "model_id", None),
                    "payload": payload,
                    "response": response_data,
                    "error": error,
                    "messages_sha256_12": sig,
                },
                f,
                indent=2,
                default=str,
            )
    except (OSError, TypeError, ValueError) as e:  # debug aid only — never fail the request
        LOG.debug("%s", IkaDebugDumpError(f"_dump_api_round failed: {e}"))

