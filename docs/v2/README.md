# IkaCore v2: simplification draft

V2 allows breaking changes. Its goal is a smaller, consistent runtime with explicit
execution controls. Backward compatibility with v1 is no longer a release requirement.
This branch remains a draft; applications should migrate deliberately before adopting it.
See the [application ergonomics handoff](ergonomics-handoff.md), [usage guide](../USAGE.md),
and generated/tested [API reference](../API.md).

## First simplification pass

The initial consolidation carried two generations of behavior at once. This pass:

- Removes the 104 copied proposal files and their inventory from the working tree
  (about 39,000 lines). They were never installed runtime code; their removal reduces
  repository and review bulk. Earlier commits in this PR retain the source material.
- Uses one implementation each for Anthropic, DeepSeek, and Gemini payload building.
  Exact live/history deduplication, stage ordering, blank-text cleanup, and completion
  tool-choice behavior are now standard.
- Uses dependency-aware workflow scheduling in both synchronous and asynchronous runs.
  Removes the old recursive/wave schedulers and their unused helper interfaces.
- Preserves an agent's own task when injecting upstream context. Prepares context once
  per node and memoizes identical default summary requests within a run.
- Uses one transient-error retry policy, with provider hints and capped jittered backoff.
  Permanent HTTP failures and programming errors fail promptly. Context-limit failures
  still reach summarization recovery.
- Removes persistent per-thread async runners and their thread-local lifecycle/task-cleanup
  machinery. The sync bridge uses an owned `asyncio.run()` loop and rejects nested-loop
  use before constructing a coroutine.
- Reduces `RuntimeOptions` from 12 fields to 6. Removes v1 import aliases and deprecated
  mutation-based tool-filter helpers, and stops fixing error classes to their old pickle path.
- Removes the v1 API snapshot contract. Tests now assert the intended v2 behavior.
- Builds verification wheels from fresh source and checks their module inventory, preventing
  stale build artifacts from reintroducing deleted modules.
- Replaces chat dependency-copying with direct calls to the owning request/summary modules.
  Removes private response/tool-loop/agent aggregate facades and the runtime helper lookup fallback.
- Collapses the workflow mixin/type-contract chains. Function and module size checks remain;
  whole-class line limits that encouraged tiny inheritance layers are removed.

The merged validation below covers runtime/test revision `179b52e`, integrating
the refactor, hard synchronous deadlines and provider harness from merge `6ef53d4`.
Historical benchmarks retain their original revision labels. Version 2.0.0 is unreleased.

## Persistence and provider optimization

- Removes the legacy checkpoint table/API and duplicate agent/stage writes. Snapshots
  are the sole persistence backend; persistence remains opt-in.
- Commits each boundary's snapshot, event and requested pause status atomically on one
  scoped connection. Nested transactions use savepoints. Tool-frame metadata and run
  setup/completion are batched without holding database locks across model/tool calls.
  Snapshot reads join lineage once instead of querying once per row; SQLite durability
  defaults remain unchanged. Existing database files are left intact.
- Bounds both tool-schema caches, including repeated fresh-list cache hits. Cache eviction
  also retires object-ID fingerprints before releasing their retained tool definitions.
- Avoids eager message-to-string conversion and speeds up the common OpenAI dictionary
  path. All seven routes retain identical payloads in the 21 benchmark fixtures.
- Codex uses supplied sync/async HTTPX clients. Async SSE collection runs natively on
  the event loop, shares the sync parser, and closes its response on cancellation while
  preserving caller-owned clients. Retry responses close before backoff; HTTP retry hints
  use the same policy in both transports. No new runtime dependency is added.

## Shared execution loop

Simple agents and stages now run through one turn loop. Control-call parsing,
completion fallback, progress logging, rate-limit entry, and extension bookkeeping
share one implementation. Stage transitions, human-input pauses, and each mode's
existing extension and empty-turn policies are preserved.

The preparation mixin chain is collapsed into one component, and dispatch uses the
shared history helper. Two loop modules become `agent_execution_loop.py`; this pass
removes 301 runtime Python lines and one module, without changing the public execution
entry points or adding dependencies.

A local before/after comparison against `78047e7` matched all 42 synthetic scenarios:
returned results, model-call sequences, usage, step limits, and snapshot boundaries.
The permanent suite also checks these behaviors across both execution modes.

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
| `workflow.run(use_async=True)` means threads | Prefer blocking `run(parallel=True, concurrency=N)`. `use_async` remains a deprecated alias. Context preparation uses a separate bounded pool. Default instance workers decrease from 30 to 4. |
| `.client = caller_client` is closed on shutdown; workflow clones discard it | Direct assignment and `ProviderConfig(client=...)` now borrow clients, preserve them across clones, and never close them. `client_factory` and internally created clients are owned; each clone gets its own. |
| `WorkflowResult.history` misleadingly means output | Use `.output` for the execution result and `.transcript` for history; `.instance_outputs`/`.instance_transcripts` retain fan-out results. `.history` remains a deprecated output alias. |
| Worker errors yield missing results and downstream KeyError | `WorkflowError.errors` identifies nodes/instances and `.partial_results` preserves completed nodes. Dependents are blocked; cancellation/deadline types remain unchanged. |
| Global logging configuration and fallback stderr warnings | Named library loggers have NullHandler and never configure the root logger. `cli=False` silences presentation; configure Python logging explicitly for diagnostics. |
| Verbose default model prompt; `Stages=` spelling | Default prompt is concise. Use `PromptConfig` for style and canonical `stages=`/`.stages`; `Stages=`/`.Stages` remain compatibility aliases. |
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
| Flat `final_answer_check`, extension settings and `use_async`, and agent extension attributes | `policy=ExecutionPolicy(final_answer_checks=(check,), ...)`; validation retries and extra steps are explicit; inspect extension settings in `agent.config.policy`. |
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
resources promptly. The pool adapter uses HTTPX/httpcore 1.x internals and is tested
with HTTPX 0.27.0 and the current supported version.

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

## Completed simplification candidates

The configuration, memory, presentation, mixin, and validation candidates are implemented:

- Frozen responsibility-specific configuration groups retain the flat common fields.
  `AgentRunState` holds live transcript, usage, budget and replay state.
- The public facade composes explicit preparation, turn loop, execution, policy, tools,
  controls, chat, lifecycle and snapshot owners; aggregate agent mixins are removed.
- Runtime emits structured scoped events. CLI, usage and file presentation are optional
  subscribers, with no process-global sink in the runtime path. Empty publishers skip
  event allocation. Legacy direct CLI APIs remain separate presentation utilities.
- IkaMem remains shipped as an explicit `MemoryTools` integration using ordinary agent
  or stage tools; no core/model dependency on IkaMem internals or global backend remains.
- `ExecutionPolicy` expresses final checks, validation retries and automatic extensions.
  Defaults preserve downstream final-answer checks and step-extension behavior.

The downstream sample contained nine directly identifiable constructor calls across
three projects. Dynamic calls/subclasses were not fully captured; common calls remain
one-liners, and this sample was not used to justify feature removal. IkaGeneral remains
outside the active runtime scope. Snapshots from `dc9efbd` are covered by a real saved
post-model fixture: resume consumes its response with no network or duplicate usage.

The merged tree passes **973 tests** (one opt-in live smoke test skipped), all coverage
floors and **42/42 fresh execution-parity traces** against `dc9efbd`. See the
[merged validation report](merged-validation.md) for final checks and the historical
[refactor report](refactor-validation.md) for its constructor migration and benchmarks.

## Checks

```bash
pip install -e ".[test,dev]"
IKACORE_LIVE_PROVIDER_TESTS=0 python scripts/quality.py
```

The suite covers request/tool controls, provider payloads, retries and context recovery,
workflow dependency/instance behavior, persistence, and replay. Live providers and
application workflows are separate release checks. Historical credential cleanup is
also separate: removing source copies or branches does not purge Git objects, PR refs,
caches, forks, or other clones.

## Live validation status

The latest Codex matrix, recorded on **2026-10-18 UTC**, used the installed 2.0.0 wheel
at runtime revision `179b52e` with `gpt-5.6-sol`: **14 passed, 0 failed, 1 skipped**.
It covers sync/async text and tools, simple/two-stage agents, fresh-process resume,
cancellation, SSE, usage, expired deadlines, synthetic retries, compaction and workflow
HITL. Resume used no network and preserved usage; cancellation closed the response while
keeping the caller's client open. `reasoning_tools` skipped with `reasoning_not_exposed`;
reasoning replay remains unverified. Every case was bounded to 60 seconds. The historical
8/8 results at `1f2bab0` and `183ce24` used the narrower original harness.

These results cover one model and account. OpenAI API Chat/Responses, Anthropic,
DeepSeek, Gemini, OpenRouter and Z.AI remain unverified live in this review. Existing
offline coverage does not establish live compatibility. Reproduce the Codex run with
[the live-check command](../../CONTRIBUTING.md#live-provider-validation).

## Measured quality and footprint

The final runtime/test revision `179b52e` on CPython 3.13.15 / HTTPX 0.28.1 passes
**973 tests**, with **one opt-in live smoke test skipped**: **96.39% line coverage**,
**88.20% branch coverage**, and **94.62% combined coverage**. Ruff, Pyright
(zero errors), compilation, fresh-wheel inventory and isolated imports also pass.
The coverage matrix assigns all **122 runtime modules** to **16 areas** and passes
**119 required area/scenario checks** (117 unique selectors), with zero gate failures.
Coverage excludes 194 lines. All global and per-area floors remain unchanged.

The fresh `ikacore-2.0.0-py3-none-any.whl` contains the final README and runtime at
`179b52e`: **212.85 KiB** compressed (217,960 bytes), **689.76 KiB** unpacked
(706,313 bytes). The runtime retains **six settings**, one direct dependency (`httpx`)
and 17,291 physical Python lines across the three packages (including comments/blanks).
The 60-case HTTP/HTTPS deadline/ownership suite passed ten consecutive rounds
(600 checks). All six Python 3.11.16 / 3.12.14 / 3.13.15 × HTTPX 0.27.0 / 0.28.1 cells
passed 967 tests with seven skips each: six dev-only lint fixtures plus the opt-in live
smoke entry point. All runtime tests ran; the main dev environment exercised the lint
fixtures. Commands, exact revisions and live results are in [merged-validation.md](merged-validation.md).

The following historical benchmark results at `183ce24` have not been remeasured after
the merged fixes. They use an installed runtime-only wheel on macOS arm64 / CPython
3.11.16, with HTTPX 0.28.1. The wheel hash and raw samples are recorded in the generated
benchmark artifact; development dependencies are excluded.

| Measurement | Result |
| --- | ---: |
| Wheel download | 202.8 KiB |
| Unpacked wheel content | 661.3 KiB |
| Installed runtime + dependencies + bytecode | 5.37 MiB |
| Direct runtime dependency | 1 (`httpx`); 7 dependency distributions total |
| Public imports / full runtime import | 3.39 / 33.16 ms |
| Runtime / Python-worker / HTTPX-only peak RSS | 32.50 / 19.36 / 27.16 MiB |
| Agent construction and shutdown | 0.0042 ms |
| One-step agent, sync / async bridge | 0.250 / 0.679 ms |
| Same agent with SQLite snapshots | 7.53 ms (before persistence consolidation: 11.81 ms) |
| 4 KiB snapshot append | 0.477 ms |
| Pure four-node workflow, sequential / parallel | 0.008 / 0.254 ms |

Medians use nine fresh-process samples and 100 warm operations per sample. The snapshot
fixture is about 36% faster than the measurement before persistence consolidation on this machine.
Imports have a warmed filesystem cache; RSS includes Python and the measurement worker.
The installed-size figure counts the complete runtime-only site-packages directory,
including compiled bytecode that some installers omit from distribution RECORD files.
Real provider/model latency is excluded. The first 1,000 repeated runs retained about
1.08 MiB of traced allocations; the next 1,000 changed retained allocations by
-46 KiB. This short soak does not establish leak freedom for every workload.

For the execution-loop refactor (`78047e7` → `183ce24`) specifically, seven alternating before/after batches
of 1,000 operations measured sync **0.2193 → 0.2194 ms** and async bridge
**0.6582 → 0.6564 ms**. This longer paired run reduces ordering noise in the short
samples above and supports treating execution latency as unchanged for this fixture.

### Provider optimization comparison (`2a1ccaf` → `78047e7`)

Payload construction for the 1,000-message / ten-tool synthetic fixture, in microseconds:

| Provider route | Before | After |
| --- | ---: | ---: |
| openai | 168.7 | 126.6 |
| openai_responses | 136.9 | 142.8 |
| anthropic | 124.5 | 127.2 |
| deepseek | 33.1 | 33.3 |
| gemini | 441.3 | 186.4 |
| openrouter | 172.9 | 128.8 |
| codex | 287.2 | 294.4 |

Seven batches of 300 warmed builds were timed separately from JSON encoding. All 21
payload checksums match the pre-optimization runtime. Gemini improves about 58%; OpenAI
Chat/OpenRouter about 25%. Other routes are roughly unchanged (up to 4% slower in this
run); these small differences are not evidence of a speedup. Codex's client reuse and
native async streaming are covered by transport tests rather than these builder timings.
Network/TLS latency, billing, live model behavior and summary quality remain unmeasured.

CI publishes package/runtime and all-provider reports. See the
[reproduction commands](../../CONTRIBUTING.md#reproducing-the-offline-benchmarks).
Persistence still dominates this trivial local agent fixture, and the runtime retains
architectural complexity. Further removal should follow observed use and product needs.

### Local coverage snapshot (`179b52e`)

| Area | Modules | Lines | Branches | Floors L/B | Tests observed | Excluded lines |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Public API / package imports | 4 | 97.89% | 90.00% | 95/90% | 301 | 20 |
| Request controls / cancellation | 6 | 99.63% | 93.75% | 95/90% | 369 | 0 |
| HTTP ownership / TLS / connection reuse | 5 | 97.76% | 95.00% | 95/90% | 285 | 2 |
| Retries / request errors | 2 | 87.66% | 78.57% | 85/75% | 221 | 0 |
| Tools / schemas / output bounds | 11 | 96.13% | 90.83% | 95/85% | 311 | 4 |
| Snapshot storage / privacy / replay rules | 8 | 94.89% | 91.13% | 90/90% | 109 | 0 |
| Agent snapshots / human resume | 6 | 99.46% | 94.23% | 95/90% | 132 | 2 |
| Workflow snapshots / selective replay | 3 | 98.65% | 92.19% | 95/85% | 57 | 0 |
| Workflows / fan-in / instance isolation | 9 | 98.28% | 93.33% | 95/90% | 109 | 66 |
| Memory integration | 10 | 93.75% | 76.67% | 90/75% | 27 | 10 |
| Logging / CLI | 5 | 97.86% | 88.75% | 95/85% | 424 | 71 |
| Agent runtime / stages | 16 | 98.18% | 87.22% | 95/80% | 463 | 19 |
| Summarization / compaction | 1 | 97.81% | 86.54% | 95/80% | 35 | 0 |
| Codex auth / streaming / payloads | 7 | 95.69% | 90.87% | 90/80% | 153 | 0 |
| Provider payloads / metadata / history | 19 | 94.75% | 84.98% | 90/80% | 190 | 0 |
| Chat loop / accounting / human interruption | 10 | 95.99% | 86.32% | 90/80% | 98 | 0 |

The generated matrix includes required scenario outcomes, exact uncovered locations,
and known gaps. CI artifacts describe each checked revision.
