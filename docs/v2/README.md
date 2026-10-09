# IkaCore v2 draft

The core runtime changes are integrated on `ikacore-v2`. Existing defaults remain
in place. This is a development draft, not a released major version or a guarantee
that every downstream application has been exercised.

## Existing projects

Existing imports, positional arguments, default provider routing, retry timing,
workflow behavior, and result shapes remain supported. The legacy `CheckpointStore`
API and independently created checkpoint databases remain readable and writable.
`enable_snapshots()` creates a separate runtime session; it does not enable or
replace an agent's legacy checkpoint configuration.

The draft adds optional keyword arguments to `IkaTools` and `ToolArgs`. It also
restores the legacy `IkaModel.google` and `IkaCore.agents.summarise_message_history`
import aliases. Optional embedded-resource loading now passes the type checker.

## Request controls and tool limits

```python
from IkaModel import request_controls

with request_controls(timeout=120, max_retries=2, tool_result_max_chars=16000):
    result = agent.execution()
```

`timeout` is an execution deadline shared by nested requests, retry waits, and
workers. `cancel_checker` accepts a callable such as `threading.Event.is_set`.
`abort_registrar` accepts a function that registers a close/cancel callback and
returns an optional unregister callback. Cancellation and deadline errors belong
to the existing API-error family and propagate through summarization and tools.

Controls are scoped with context variables and restored on exit. Workers inherit
explicit controls, runtime options, and snapshot hooks. A nested deadline cannot
extend its parent's deadline. `max_retries` follows the existing attempt-count
meaning and is clamped to at least one when explicitly overridden.

No tool-output limit is enabled by default. Explicit limits are available through
`tool_result_max_chars`, `codex_tool_output_max_chars`, or the corresponding
`IKA_TOOL_RESULT_MAX_CHARS` / `IKA_CODEX_TOOL_OUTPUT_MAX_CHARS` environment variables.
Clipping includes a marker and keeps the beginning and end within the character cap.
Tools can also supply `IkaTools(..., timeout=30)` independently of the default timeout.

Synchronous Python functions cannot be forcibly stopped by cancelling a future.
A timed-out worker may finish later. Async coroutines are cancelled cooperatively;
registered network operations are closed on cancellation/deadline. Callers should
provide cooperative cancellation for functions that perform lasting side effects.

## Agent snapshots

```python
from IkaCore import RuntimeControl

store = agent.enable_snapshots(
    "runtime-data/snapshots.db",
    RuntimeControl(pause_points={"post_model"}),
)
paused = agent.execution()
result = agent.resume_from_snapshot(paused["checkpoint_uid"])
```

Snapshots record runs, agent/stage/tool frames, histories, usage, costs, and event
sequences. Boundaries include `frame_entry`, `step_entry`, `stage_entry`,
`pre_model`, `post_model`, `pre_tool`, `post_tool`, `pre_compaction`,
`post_compaction`, `human_input`, and `agent_completed`. Pause points support
wildcards. `request_pause_at()` and `clear_pause_requests()` configure an agent's
session. Snapshot-enabled results add a `runtime` field; paused results include
`status="paused"` and a snapshot UID.

Resuming a post-model snapshot consumes its saved response without charging usage
again. Completed snapshots return their saved output. Staged human-input snapshots
accept `resume_input`. `fork_from_snapshot()` and `restart_from_frame()` create new
runs and preserve the source run. Subagents inherit the active parent run and record
a call tree.

Generator tools may yield `{"__ika_checkpoint__": {"label": "progress", "payload": {...}}}`
while snapshot hooks are active. A mid-tool snapshot records progress; it cannot
serialize a Python stack or continue an arbitrary generator. Such replay requires
`allow_unsafe_replay=True` and restarts execution. Tool replay defaults to denied;
`replay_policy="allow"` is an explicit caller assertion, not an inferred guarantee.
Replay assessment includes nested and partially completed tool frames.

New runtime JSON masks known credential fields/token formats and credentials in
URLs. Snapshots whose state was redacted are refused for replay by default.
This filtering cannot identify every confidential value. Keep databases private.
Snapshot IDs are immutable; predecessors must exist in the same run. Legacy
checkpoint serialization is unchanged.

## Runtime policies

```python
from IkaModel import runtime_options

with runtime_options(
    optimize_provider_payloads=True,
    modern_retries=True,
    compact_at_fraction=0.7,
):
    result = agent.execution()
```

All these behavior changes require an explicit opt-in:

| Option | Effect |
| --- | --- |
| `optimize_provider_payloads` | Deduplicate exact persisted/live tool rounds; order stage and human-input turns; remove blank Anthropic text; avoid forcing completion tools. |
| `modern_retries` | Retry transient statuses, honor provider hints, use capped backoff with jitter, and fail promptly for permanent/programming errors. |
| `shared_ssl_context` | Reuse verified SSL contexts while tracking CA environment/file changes. `clear_ssl_context_cache()` explicitly invalidates the cache. |
| `reuse_connections` | Reuse execution-owned synchronous HTTPX clients, isolated by URL, headers, and proxy/SSL environment. Close them at scope exit. |
| `persistent_async_runner` | Reuse one event loop per worker thread, copy context for each call, and cancel leftover tasks between calls. Refuse nested synchronous calls before touching an active loop. |
| `dataflow_workflows` | Schedule a node after its own dependencies complete; prepare context once per node; honor synchronous instances and fan-in. |
| `parallel_context_preparation` | Allow concurrent context preparation when dataflow scheduling is enabled; custom hooks must support concurrency. |
| `preserve_workflow_prompts` | Inject context alongside the agent's task and stage prompt; replace prior injected context on a new run. |
| `summary_context_threshold` | Pass short workflow context through instead of invoking a summarizer; uses an approximate character-based token estimate. |
| `compact_at_fraction` | Change the existing proactive history-summary threshold; default remains 0.8. |
| `recover_truncated_control_calls` | Recover structurally truncated JSON for completion/stage control tools. Ordinary tool arguments remain strict. |

`get_context_usage()` in `IkaModel.summarization` reports token accounting and warning
levels without modifying history or making a request. Workflow summary memoization
is run-scoped and only applies to the default compressor with identical inputs and
configuration; custom compression hooks are not memoized.

## Workflow snapshots

`workflow.enable_snapshots(path, control)` enables dependency-aware execution for
that workflow. `workflow.run(...)` retains its result mapping. Inspect
`workflow.snapshot_runtime_info()` for the run, latest snapshot, and paused state.
Completed instance results can be reused by `workflow.resume_from_snapshot(uid)`.
`RuntimeControl(parallel_replay={"node": [instance_id]})` selects instances to rerun;
all their downstream results are invalidated. Disabling
`reuse_historical_parallel_results` reruns all instances. Unsafe tool replay requires
an explicit override. Changed workflow tasks/graphs or invalid instance IDs are
refused. Snapshot replay assumes the same configured tool implementations.

Workflow pauses preserve completed instances. Already-running sibling work may
finish before the scheduler returns; pauses are not atomic transactions across
threads. Use request cancellation controls when all workers must stop cooperatively.
Use `run()` for workflow snapshot sessions; direct `run_async()` is the legacy
scheduler entry point and does not start a workflow snapshot session.

## Integration decisions and validation

The [source inventory](../../development/v2/source/inventory.json) records adapted,
retained, and excluded proposals. The IkaGeneral experiment remains isolated review
material. Global async transport caches, dropping earlier summary context, automatic
deletion of oversized user input, and application-specific guardrail prompts were
excluded to preserve ownership and task semantics.

Tests cover default API/checkpoint compatibility, deterministic cancellation and
retry lifecycles, provider payloads, tool limits, snapshots, replay, and workflows.
Consumer-only static imports/call shapes were checked in three dependent projects;
no new regressions were found. This is not an end-to-end application or live-provider
test. No dependent security-agent workflows were executed.

```bash
pip install -e ".[test,dev]"
IKACORE_LIVE_PROVIDER_TESTS=0 python scripts/quality.py
```

Branch deletion does not erase old Git objects, PR references, caches, forks, or
other clones. Credential rotation and historical-data cleanup remain separate from
this integration. No history rewrite or merge to `main` is part of this draft.
