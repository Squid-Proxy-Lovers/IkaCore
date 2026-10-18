# IkaCore 2.0 migration

IkaCore 2.0.0 is unreleased and intentionally breaks v1 compatibility. New applications
should start with the [usage guide](../USAGE.md) and [API reference](../API.md).
Existing applications can use the migration table below to adopt explicit configuration,
conversation turns and durable workflows.

The runtime uses one execution loop, frozen configuration and mutable run state.
Snapshots provide pause, resume, fork and human approval; memory is supplied through
ordinary tools. See [provider reasoning](../providers/reasoning.md) and the concise
[release validation](validation/release-2.0.0.md) for supported contracts and limits.

## Breaking changes and migration

| Previous call or behavior | V2 behavior / migration |
| --- | --- |
| Positional construction of grouped config/result dataclasses | Use keyword fields when migrating: new context/validation/transcript fields change positional bindings. In particular, construct `WorkflowResult(..., output=..., child_summaries=...)` rather than `history=` or a positional child-summary mapping. |
| Validator receives an empty history at construction; docstring/return annotation required | Construction checks callability/signature only. A check receives `IkaCore.AgentResult` on completion and returns `bool`; runtime rejects non-bools. Docstrings are optional correction feedback. |
| Staged completion skips final checks; simple correction reruns application tools | Both validate final answers. Bounded correction passes use the transcript and only `agent_end`, without repeating application side effects. |
| Invalid output silently returned after retry exhaustion | Inspect `result["validation"]` (`status`, `attempts`, `failed_checks`); opt into `ExecutionPolicy(fail_closed=True)` and catch `FinalAnswerValidationError.result`. |
| Assigning `agent.prompt` leaves stale first input | Assignment starts a fresh task and clears transcript/pending state. Use `agent.send(message)` for an appended conversation turn and fresh per-turn steps. |
| Final tool completion may omit answer text from history | User turns and final answers are recorded in order; `.transcript` is a defensive, possibly compacted replay transcript. |
| Completed-snapshot restore returns before loading conversation; ignores an answer | Restores transcript/usage/counters and returns saved output without a request. Use `send()` next. `resume_input` on completed agent snapshots now raises `ValueError`. |
| Agent fork has no answer argument; workflow fork requires two calls | `agent.fork_from_snapshot(id, resume_input=...)` and `workflow.fork_from_snapshot(id, resume_input=...)` fork and continue independently. |
| `agent.context_budget = N` is outside grouped config | Prefer `limits=ExecutionLimits(context_budget=N)` in tokens; `compaction_count` and `compaction` events are observable and snapshot-persisted. The direct attribute still works. |
| `workflow.run(use_async=True)` means threads | Use blocking `run(parallel=True, concurrency=N)`. Workflow `use_async` and `run_async()` are removed. Context preparation uses a separate bounded pool. Default instance workers decrease from 30 to 4. |
| `.client = caller_client` is closed on shutdown; workflow clones discard it | Direct assignment and `ProviderConfig(client=...)` now borrow clients, preserve them across clones, and never close them. `client_factory` and internally created clients are owned; each clone gets its own. |
| `WorkflowResult.history` misleadingly means output | Use `.output` for the execution result and `.transcript` for history; `.instance_outputs`/`.instance_transcripts` retain fan-out results. The `.history` alias is removed. |
| Worker errors yield missing results and downstream KeyError | `WorkflowError.errors` identifies nodes/instances and `.partial_results` preserves completed nodes. Dependents are blocked; cancellation/deadline types remain unchanged. |
| Global logging configuration and fallback stderr warnings | Named library loggers have NullHandler and never configure the root logger. `cli=False` silences presentation; configure Python logging explicitly for diagnostics. |
| Verbose default model prompt; `Stages=` spelling | Default prompt is concise. Use `PromptConfig` for style and canonical `stages=`/`.stages`; `Stages=`/`.Stages` are removed. |
| Small apps must override many unrestricted defaults; mocks patch internals | Use `ProviderConfig.bounded()`, `ExecutionLimits.bounded()`, `ExecutionPolicy.bounded()`, plus `request_controls(timeout=...)`. Use public `IkaCore.testing.MockProvider` for native fixtures. |
| Agent `memory`, `memory_access`, backend fields/helpers and `MemoryConfig` | Construct `IkaMem.agent_tools.MemoryTools(name, short_term=..., long_term=...)`; pass `.tools(access=...)` using ordinary agent/stage `tools`. |
| Stage `memory_access`, `long_term_filter` | Select `.tools(access=..., long_term_filter=...)` for that stage. |
| `IkaModel.init_global_long_term_memory` / `get_global_long_term_memory` | Construct and explicitly share an `LTMemory` backend. No global backend is read. |
| Agent mixin subclassing/private helpers and `agent_execution` / `agent_helpers` imports | Use the public facade or the explicit component owners listed in `docs/indepth.md`; private helpers no longer appear on the agent. |
| `agent_tool_conversion` / `agent_tool_execution` / `agent_tool_schema` | Tool ownership is consolidated in `agent_tools`. |
| Global CLI sink/stdout setters controlling runtime output | Use `output=OutputConfig(cli=False, subscribers=(callback,))` on an agent/workflow, or `output_scope(RuntimeOutput(...))` for standalone model/tools. Runtime emits structured events to optional CLI, usage and file adapters. Legacy setters only affect direct legacy CLI calls. |
| Required agent description | Description is optional (defaults to name); `IkaBaseAgent("helper", prompt="task", model_id=..., api_key=...)` is supported. |
| Flat `api_url`, `max_tokens`, `temperature`, `use_responses_api`, `reasoning_effort` | `provider=ProviderConfig(...)` |
| Flat `system_prompt`, `start_prompt`, `end_prompt` | `prompts=PromptConfig(...)` |
| Flat `maxsteps`, `step_timeout`, tool budgets and rate limits | `limits=ExecutionLimits(...)` |
| Flat `final_answer_check`, extension settings and `use_async`, and agent extension attributes | `policy=ExecutionPolicy(final_answer_checks=(check,), async_requests=True, ...)`; the old `use_async` name is removed. Validation retries and extra steps are explicit; inspect extension settings in `agent.config.policy`. |
| Flat `subagents`, `next_agent` | `topology=AgentTopology(subagents=(child,), next_agent=...)` |
| Flat logging/usage/summary settings | `output=OutputConfig(...)` |
| Configuration mixed with live counters/history | `agent.config` is frozen; `agent.state` stores the current run's budgets, transcript, usage and replay state. |
| `CheckpointStore`, `load_checkpoint()`, `checkpoint=True`, `checkpoint_db_path=...`, stage `checkpoint` | Removed. Call `agent.enable_snapshots(path)` or `workflow.enable_snapshots(path)`. V1 checkpoint databases are not converted. |
| `agent.execution(checkpoint_uid=...)` / returned `checkpoint_uid` | Call `agent.resume_from_snapshot(snapshot_id, resume_input=...)`; returned IDs are `snapshot_id`. |
| `optimize_provider_payloads`, `modern_retries`, `dataflow_workflows`, `preserve_workflow_prompts` | Remove these options; their reviewed behavior is now standard. |
| `persistent_async_runner`, `close_thread_runner()` | Removed. Each synchronous async call owns its event loop. |
| `retry_initial_seconds` in `runtime_options()` | Use the request's `wait_seconds` argument, now the initial jittered-backoff delay, default 1 second. |
| Retrying every HTTP/programming error | Retry transient statuses (408/409/429/5xx) and transport failures. Permanent errors propagate. |
| Required `agent_end`/stage controls force a tool call | Completion/stage controls no longer force tool selection. Explicit provider tool choice remains available. |
| Synchronous workflows mutate the template agent | Every instance runs on a clone. Inspect returned results and snapshots for execution state. |
| Old workflow traversal and partial dependency context | A node runs after all its dependencies. Cyclic/unreachable prerequisites are refused before execution. |
| Upstream context replaces a task or is repeatedly prepended | Context is refreshed alongside the agent's own task and stage instructions. |
| `IkaModel.google` | Import from `IkaModel.gemini.google`. |
| `IkaCore.agents.summarise_message_history` | Import from `IkaModel.summarization`. |
| `_apply_tools_filter_for_payload` / `_restore_tools_after_payload` | Use `model_for_payload()`; it returns a filtered copy. |
| Private helpers imported through `agent_runtime`, `response_interface`, `chat_tool_loop`, or runtime lookup forwarding | Import the owning component directly. Tests should patch request functions in `IkaModel.request_interface` and summaries in `IkaModel.summarization`. |
| Provider error classes pickled under `request_interface` | Classes live in `IkaModel.api_errors`; old serialized exception objects are unsupported. |

Standard JSON-request errors expose status instead of embedding provider response bodies. Code matching old exception text should use
the error type/status instead.

## Retained controls

```python
from IkaModel import request_controls, runtime_options

with request_controls(timeout=120, max_retries=2, tool_result_max_chars=16000):
    with runtime_options(compact_at_fraction=0.7):
        result = agent.execution()
```

Request controls provide shared deadlines, cancellation checks, abort registration,
retry-count overrides, and optional general/Codex output caps. Nested deadlines cannot
extend their parent. `IkaTools(..., timeout=30)` sets an individual tool timeout.
Synchronous functions must cooperate with cancellation; a timed-out worker may finish later.

Synchronous JSON posts and Codex SSE requests with a deadline or cancellation scope
run in a context-preserving worker. The caller's wait uses the absolute deadline,
so trickled headers/bodies, stalled TLS handshakes, and blocked hooks cannot extend
it. Abort registration wakes the caller immediately; a cancellation checker without
a registrar is polled at most every 10 ms. These are wall-clock bounds subject to
normal OS/Python thread scheduling, rather than renewed per-I/O timeout budgets.
Async requests retain native task cancellation throughout the request.

For HTTPX's built-in transport, each controlled synchronous request uses a private
httpcore pool with the caller's TLS, proxy/mount, auth, cookie, hook, redirect, and
protocol settings. No connection enters the caller's pool. Abort shuts down only
duplicated descriptors from that private pool, preserving socket identity even
during TLS wrapping or after a response hook consumes the body. Stream cleanup
callbacks also cover connection/TLS hand-offs before httpcore attaches the stream
to its connection. The worker closes its client, streams, pool, and descriptors. Caller-owned clients/transports remain open;
concurrent and subsequently pooled requests are unaffected. Without a lifecycle
scope the existing direct transport path is unchanged and creates no worker.

The caller also returns on time during blocking DNS, pre-connect system calls,
client initialization, or arbitrary custom transport/hook code. Python cannot
forcibly stop those operations: their daemon worker and resources remain until
they return. Code that prevents Python thread scheduling, such as a C extension
holding the GIL, can also delay observation. Custom transports are borrowed, never closed or inspected
for sockets; response cleanup is attempted separately and must cooperate to reclaim
resources promptly. The pool adapter uses HTTPX/httpcore internals. Supported dependencies are `httpx>=0.27,<0.29` and
`httpcore==1.0.9`; missing pool internals raise an actionable error before transport use.

The remaining runtime settings are `shared_ssl_context`, `reuse_connections`,
`parallel_context_preparation`, `recover_truncated_control_calls`, `compact_at_fraction`,
and `summary_context_threshold`. These configure resource ownership, concurrency, and
context policy rather than switching between v1 and v2 implementations.

## Retained snapshots

```python
from IkaCore import RuntimeControl

store = agent.enable_snapshots("runtime-data/snapshots.db", RuntimeControl(pause_points={"post_model"}))
paused = agent.execution()
result = agent.resume_from_snapshot(paused["snapshot_id"])
```

Agent snapshots record run/stage/tool/subagent frames, history, usage, and events.
Nested agents own their snapshot frames; returning to the parent restores its execution
context. Post-model resume consumes the saved response without counting usage again;
completed snapshots return saved output. Fork/restart creates a new run. Resuming a
historical snapshot checks subsequent tool effects, including nested calls, and requires
`allow_unsafe_replay=True` before repeating unsafe effects. Mid-tool replay also requires
this explicit approval; replay cannot restore a Python stack or undo external effects.

Staged human-input resume accepts `resume_input`, consumes the saved transcript once,
and appends the answer before continuing. Snapshots persist consumed steps, the current
step limit, extension counts, and the remaining overall stage budget. Mid-stage resume
preserves those limits even in a new agent and does not charge consumed steps twice to
later stages. Exhausted human-input resumes apply extension or forced finalization
before completing the stage. Earlier v2 snapshots without `loop_state` use their
remaining budgets without a database migration. Fresh executions clear replay state,
including after a failed resume.

Workflow snapshots are enabled with `workflow.enable_snapshots(path, control)` and
started with `workflow.run(...)`. `snapshot_runtime_info()` exposes the last snapshot.
With workflow snapshots enabled, an instance returning `paused` or `awaiting_user_input`
pauses the workflow and blocks its dependents in both sync and async runs. Interrupted
outputs are not completed results. The workflow snapshot retains interrupted child
snapshot IDs; `snapshot_runtime_info()["interrupted_instances"]` maps `node:instance`
keys (for example, `review:0`) to those IDs. Answer a single interrupted instance with
`workflow.resume_from_snapshot(workflow_snapshot_id, resume_input="YES")`. For multiple
instances, pass `resume_input={"review:0": "YES", "review:1": "Revise"}`; omitted
instances remain paused. Resuming without an answer preserves pending human questions.
If a child was already resumed separately and completed, workflow resume reuses its
output and continues dependents. Completed siblings are retained; selective replay and
interrupted instances invalidate downstream cached results.
Nested and unfinished tool effects are checked before replay. Snapshot JSON filters
known credential patterns and redacted state is refused for replay by default. Filtering
cannot identify all private content, so keep runtime databases private.
