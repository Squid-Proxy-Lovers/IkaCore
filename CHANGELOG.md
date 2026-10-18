# Changelog

## 2.0.0 (Unreleased)

V2 intentionally breaks compatibility with v1. See the
[migration guide](docs/v2/README.md#breaking-changes-and-migration).

- Removed `CheckpointStore`, `load_checkpoint()`, constructor `checkpoint` and
  `checkpoint_db_path`, stage `checkpoint`, and `execution(checkpoint_uid=...)`.
  Enable the SQLite `SnapshotStore` with `enable_snapshots()` and resume using
  `resume_from_snapshot(snapshot_id, resume_input=...)`. V1 checkpoint databases
  are left intact and are not converted. Finish old runs with v1 before migrating.
- Snapshots record runs, nested frames, immutable state and events; boundaries commit
  atomically. Agent resume, fork/restart, HITL interrupts and workflow selective replay
  use this one backend. Replay cannot restore Python stacks or undo tool side effects;
  unsafe replay requires an explicit opt-in.
- Removed runtime settings `optimize_provider_payloads`, `modern_retries`,
  `dataflow_workflows`, `preserve_workflow_prompts`, `persistent_async_runner` and
  `retry_initial_seconds`, plus `close_thread_runner()`. Use request `wait_seconds`
  for the initial retry delay (default one second).
- Provider history replay and transient-error retries are now standard. Permanent
  HTTP/programming failures propagate. OpenAI uses Responses by default; completion
  controls no longer force tool selection. Sync workflows clone template agents and
  both workflow modes wait for dependencies while preserving each agent's task.
- Removed v1 import/private forwarding facades, the `IkaModel.base.TOKENMAX_MAPPING`
  alias (use `IkaModel.model_metadata`), unused `IkaCheckpointError`, and the unused
  Gemini payload loader (import `gemini_fill_payload` from `IkaModel.gemini.google`). Public packages report the
  installed `ikacore` version and ship `py.typed` markers.
- Examples use environment credentials and avoid printing bearer fragments. Documentation
  now covers functional memory setup, current snapshot APIs and offline validation.

### Agent configuration

- Agent options are frozen `ProviderConfig`, `PromptConfig`, `ExecutionLimits`,
  `ExecutionPolicy`, `AgentTopology`, and `OutputConfig` values. Pass them using
  `provider=`, `prompts=`, `limits=`, `policy=`, `topology=`, and `output=`.
  The corresponding flat constructor keywords are removed. Name, description,
  prompt, model_id, api_key, Stages, and tools remain flat.
- `agent.config` retains configured defaults; `agent.state` owns mutable run budgets,
  transcripts, usage, counters and replay state. Extensions do not mutate config.
  Final checks and automatic continuation settings are one explicit execution policy;
  defaults preserve three validation retries and the existing extension behavior.

### Optional memory integration

- Removed agent `memory` / `memory_access`, `MemoryConfig`, backend attributes and
  memory helper methods, stage `memory_access` / `long_term_filter`, and
  `IkaModel.init_global_long_term_memory` / `get_global_long_term_memory`.
  Use `IkaMem.agent_tools.MemoryTools(name, short_term=..., long_term=...).tools()`
  as ordinary agent or stage tools. Storage, access, filters and sharing are explicit.
  Core/model packages do not import IkaMem. Search tools allow replay; saves deny it.

### Runtime ownership

- Removed the agent mixin tree and aggregate `agent_execution` / `agent_helpers`
  modules. The facade composes preparation, turn loop, execution policy, tools,
  control parsing, chat/output, lifecycle and snapshot components. Public entry
  points remain; private helpers and tests patch the owning component directly.
- Merged `agent_tool_conversion`, `agent_tool_execution` and `agent_tool_schema`
  into `agent_tools`; pure history/validation helpers remain in `agent_helper_support`.

### Structured output events

- Runtime presentation uses scoped `RuntimeOutput` / `RuntimeEvent` publishers with
  optional CLI, file and usage subscribers. Configure agent/workflow `output=` or use
  `output_scope()` for standalone calls. Runtime output no longer follows legacy
  global CLI sink/stdout setters; those remain for direct presentation utility users.
  CLI boxes and file formats retain their defaults. Empty publishers skip event allocation.
- Worker contexts propagate output subscriptions; workflow instances share CLI buffering.
  Subscriber exceptions propagate, and nested/exceptional scopes restore their parent.
- Agent description is optional (defaults to name), keeping common calls short.
- A saved snapshot fixture written by `dc9efbd` resumes without a provider call or
  duplicate usage. Its former memory-access state is ignored; explicit tools configure memory.

### Refactor verification

- Permanent 42-scenario execution parity checks compare the grouped API with `dc9efbd`.
  Components remain lazy and frozen options reusable. Coverage floors are unchanged.
- The installed-wheel live harness uses grouped provider, limit, policy and output
  configuration. Its offline probes cover agent construction for all eight routes,
  simple/staged completion, fresh-process resume, compaction and workflow HITL.
  Controlled JSON/SSE integration tests verify context variables, output subscribers,
  snapshot hooks, worker-raised pauses and post-model replay without duplicate usage.
- Validation includes deterministic provider, retry, execution, workflow, snapshot,
  memory and cancellation tests; lint/type/coverage gates; wheel checks; repeated
  real-socket deadlines; and Python 3.11–3.13 / HTTPX compatibility checks.
  See [the merged validation report](docs/v2/merged-validation.md) for exact revisions,
  measurements and live Codex results (14 passed, 1 `reasoning_not_exposed` skip with
  `gpt-5.6-sol` at `179b52e`). Other providers/models and long-running application
  behavior require separate live validation.

### Fixed

- Preserve context variables in the background response-cleanup thread after a
  controlled request expires. Custom stream cleanup retains its output subscribers,
  workflow instance ID, runtime options and request controls after the caller returns.
- Preserve zero remaining steps in legacy staged HITL snapshots without `loop_state`
  so sync and async resumes cannot regain a model turn.

- Invalidate interrupted downstream transcripts and their old answers when upstream
  workflow instances rerun. Restart with fresh inputs and enforce replay-safety checks.

- Clone interrupted workflow child continuations into forked runs, remapping their frames
  and snapshots, and reject cross-run child snapshots before routing human answers.

- Run exhausted HITL resumes through stage extension or forced finalization before
  completing the stage, so a last-step question cannot bypass the human answer.
- Persist interrupted workflow instances' snapshot IDs and route workflow `resume_input`
  answers to those children. Reuse separately completed child outputs, retain completed
  siblings, and continue dependents in sync and parallel workflows.
- Keep remaining step budgets for older v2 snapshots without `loop_state`, including
  the consumed allowance recorded by the current stage's entry boundary.
- Enforce absolute wall-clock deadlines and prompt cancellation for synchronous JSON
  posts and Codex SSE, including trickled headers/bodies and stalled TLS handshakes.
  Controlled requests run on a context-preserving worker with private HTTPX/httpcore
  pools that retain caller configuration. Abort shuts down only duplicated sockets
  owned by those pools; caller clients and their concurrent/pooled requests stay
  untouched, including when response hooks preload bodies. The uncontrolled path
  creates no worker. Check-only cancellation is polled every 10 ms. Blocking DNS,
  pre-connect system calls, and uncooperative custom code may finish and release
  worker resources later without extending the caller's wait; wall-clock return
  remains subject to normal thread scheduling.
- Require explicit `allow_unsafe_replay=True` approval before historical snapshot
  resume repeats unsafe tool effects, including nested calls.
- Pause snapshot-enabled workflows when an agent waits for human input, blocking its
  dependents in sync and async runs. Do not reuse interrupted outputs as completed
  results; invalidate their downstream cached results during replay.
- Keep nested-agent snapshots in the child's frame and restore the parent's execution
  context when the child returns.
- Restore the saved HITL transcript once and append the human answer before continuing.
- Clear stale replay state before fresh executions, including after failed resumes.
- Preserve consumed steps, step limits, extension counts and remaining overall stage
  budgets across resume, including in a new agent. Avoid charging prior steps twice to
  later stages.
- Close SQLite connections when snapshot setup fails.
