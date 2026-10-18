# Changelog

## 2.0.0 (Unreleased)

IkaCore 2.0 intentionally breaks compatibility with v1. Start with the
[usage guide](docs/USAGE.md), [API reference](docs/API.md) and
[migration guide](docs/v2/README.md).

### Highlights

- Build simple or staged agents with explicit frozen configuration, mutable run state,
  bounded presets and one execution loop.
- Continue conversations with send(), validate final answers through AgentResult, and
  compose dependency-aware workflows with structured failures and partial results.
- Pause, resume and fork durable snapshots, including human approval in a fresh process.
- Replay provider-native reasoning across tools, conversation turns and snapshots.

### Breaking changes

- Advanced flat constructor options move to ProviderConfig, PromptConfig, ExecutionLimits,
  ExecutionPolicy, AgentTopology and OutputConfig. Common agent fields remain direct arguments.
- Use lowercase stages throughout. Stages, workflow use_async/run_async and
  WorkflowResult.history are removed. Use run(parallel=True, concurrency=N), output
  and transcript. Agent request bridging is ExecutionPolicy(async_requests=True).
- Assigning prompt starts a fresh task; send(message) appends a turn with a fresh budget.
  Completed-snapshot restore loads the conversation and returns saved output; send()
  continues it. resume_input on a completed agent snapshot raises ValueError.
- Final checks receive AgentResult and return bool. Construction checks signatures without
  executing validators; simple and staged completion share validation. Correction passes
  expose only agent_end. fail_closed=True raises FinalAnswerValidationError on exhaustion.
- Snapshots replace v1 checkpoints. Existing checkpoint databases are left intact and are
  not converted; finish old runs with v1 before migrating.
- Memory setup is explicit through IkaMem.agent_tools.MemoryTools and ordinary tools.
  No implicit global memory backend or agent memory configuration remains.
- Configuration rejects invalid types, negative allowances, nonfinite budgets, invalid
  temperatures and zero values where a positive budget is required. Supported zero
  allowances are documented in USAGE.

### Added

- Public offline MockProvider fixtures, scoped RuntimeEvent subscribers and six runnable
  example applications covering research, code review, approval, chat, fan-in and providers.
- Atomic snapshot boundaries, nested frames, guarded replay, per-instance workflow output
  and transcripts, fresh-process continuation and fork resume_input.
- Context budgets in tokens, persisted compaction counts/events and defensive transcripts.
- Native OpenAI Responses/Codex reasoning items, Anthropic signed/redacted thinking,
  Gemini signed parts, OpenRouter reasoning details, and DeepSeek/Z.AI reasoning replay.
- Typed public lazy imports and installed-wheel consumer/type-completeness checks.
  Release checks build wheel and sdist, validate metadata, run documentation/examples/apps
  offline, check local documentation links and enforce coverage/type floors.
- Six CI cells: Python 3.11–3.13 with minimum and current supported HTTPX.

### Changed

- OpenAI uses Responses by default. Provider history replay, transient-error retries and
  dependency-aware workflow scheduling are standard behavior.
- Workflow runs clone templates, preserve each task and wait for all dependencies.
  Default worker count is four; context preparation has its own bounded pool.
- Borrowed clients remain open, including across clones. client_factory creates an owned
  client per instance; workers and delegated agents close their owned resources.
- Runtime output is scoped and uses optional CLI/file/usage subscribers. Named loggers
  never configure root logging; cli=False suppresses presentation.
- Runtime dependencies are bounded to httpx>=0.27,<0.29 and httpcore==1.0.9, with an
  actionable guard for missing private transport internals.

### Fixed

- Enforce absolute request deadlines and cancellation through JSON, SSE, TLS handshakes,
  trickled bodies, retries and hooks without interrupting caller-owned pooled requests.
  Cleanup workers preserve runtime context. Blocking DNS/custom code can finish later.
- Preserve ordered pending turns, final answers, usage and consumed step/extension budgets
  exactly once across fresh-process resume, stage transitions and validation corrections.
- Route nested and workflow human answers to interrupted children; retain completed
  siblings and invalidate stale downstream results during selective replay.
- Propagate cancellation, deadlines and worker errors with their original identity;
  WorkflowError includes node/instance details and partial results.
- Retain constraints, latest input, complete tool exchanges and reasoning during compaction;
  truncated summaries preserve history and interrupted provider answers skip partial tools.
- Apply model-aware Anthropic thinking/sampling/tool choice, Gemini dynamic thinking,
  cached/thought usage accounting, multiline Codex SSE and capped provider tool results.
- Manual secret scans inspect the working directory without accidentally scanning all history.
- Align stage_end(reason=...), provider reasoning, approval fork and silent reviewer docs
  with the implemented APIs.

### Removed

- CheckpointStore, load_checkpoint(), constructor/stage checkpoint flags, checkpoint_db_path
  and execution(checkpoint_uid=...); use SnapshotStore and snapshot APIs.
- Agent memory/memory_access, MemoryConfig, backend helpers, stage memory_access/
  long_term_filter, and IkaModel global-memory initialization/access helpers.
- Runtime switches optimize_provider_payloads, modern_retries, dataflow_workflows,
  preserve_workflow_prompts, persistent_async_runner, retry_initial_seconds and
  close_thread_runner(). Request wait_seconds supplies the initial retry delay.
- V1 private forwarding/import facades, aggregate agent mixins/helpers, TOKENMAX_MAPPING,
  unused IkaCheckpointError and the unused Gemini payload loader.

See [release validation](docs/v2/validation/release-2.0.0.md) for reproducible evidence
and limits. Historical development reports are archived outside the adoption path.
