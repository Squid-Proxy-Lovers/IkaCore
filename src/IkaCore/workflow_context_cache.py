"""Run-scoped summaries; custom compression hooks are never memoized."""

# pyright: strict

import threading
from typing import Any


def reset_context_cache(workflow: Any, initial: str | None) -> None:
    workflow._initial_context = initial
    workflow._summary_memo = {}
    workflow._summary_lock = threading.Lock()


def compress_once(workflow: Any, node: Any, contexts: list[str]) -> str:
    if not hasattr(workflow, "_summary_lock"):
        reset_context_cache(workflow, None)
    from .workflow_core import WorkflowContextMixin
    if getattr(workflow.compress_hook, "__func__", None) is not getattr(WorkflowContextMixin, "_default_compress_hook"):
        return workflow.compress_hook(contexts, node.agent)
    fields = ("model_id", "api_url", "api_key", "use_responses_api", "max_tokens", "temperature",
              "reasoning_effort", "system_prompt", "prompt", "description")
    key = (tuple(contexts), tuple(getattr(node.agent, field, None) for field in fields))
    with workflow._summary_lock:
        slot = workflow._summary_memo.setdefault(key, {"lock": threading.Lock(), "value": None})
    with slot["lock"]:
        if slot["value"] is None:
            slot["value"] = workflow.compress_hook(contexts, node.agent)
        return slot["value"]
