# IkaCore v2: simplification draft

V2 allows breaking changes. Its goal is a smaller, consistent runtime with explicit
execution controls. Backward compatibility with v1 is no longer a release requirement.
This branch remains a draft; applications should migrate deliberately before adopting it.

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

The current offline quality and wheel measurements below cover runtime/test revision
`0ec020a`. Historical benchmarks and live checks retain their original revision labels.
Live-provider and application behavior remain separate release checks.

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
| Agent `memory`, `memory_access`, backend fields/helpers and `MemoryConfig` | Construct `IkaMem.agent_tools.MemoryTools(name, short_term=..., long_term=...)`; pass `.tools(access=...)` using ordinary agent/stage `tools`. |
| Stage `memory_access`, `long_term_filter` | Select `.tools(access=..., long_term_filter=...)` for that stage. |
| `IkaModel.init_global_long_term_memory` / `get_global_long_term_memory` | Construct and explicitly share an `LTMemory` backend. No global backend is read. |
| Flat `api_url`, `max_tokens`, `temperature`, `use_responses_api`, `reasoning_effort` | `provider=ProviderConfig(...)` |
| Flat `system_prompt`, `start_prompt`, `end_prompt` | `prompts=PromptConfig(...)` |
| Flat `maxsteps`, `step_timeout`, tool budgets and rate limits | `limits=ExecutionLimits(...)` |
| Flat `final_answer_check`, extension settings and `use_async` | `policy=ExecutionPolicy(final_answer_checks=(check,), ...)`; validation retries and extra steps are explicit. |
| Flat `subagents`, `next_agent` | `topology=AgentTopology(subagents=(child,), next_agent=...)` |
| Flat logging/usage/summary settings | `output=OutputConfig(...)` |
| Configuration mixed with live counters/history | `agent.config` is frozen; `agent.state` stores the current run's budgets, transcript, usage and replay state. |
| --- | --- |
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

Synchronous HTTP posts register abort before headers, clamp connect/read/write/pool
timeouts to the remaining deadline, and close only the response or stream they own.
Caller-owned clients stay open, and cancellation never shuts down a raw pooled socket.
The timeout clamp uses HTTPX's optional [trace extension](https://www.python-httpx.org/advanced/extensions/).
Explicit cancellation of a blocked sync read may take effect only at the next clamped
I/O timeout. Deadlines bound each I/O wait rather than guaranteeing a hard wall-clock
return: DNS/TLS, trickled bodies, and custom transports can delay the next control check.
Async requests support task cancellation throughout the request.

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

## Remaining candidates for simplification

| Area | Evidence and proposed direction |
| --- | --- |
| Agent configuration/inheritance | The turn loop and preparation component are unified. `IkaBaseAgent` still accepts 35 constructor arguments; group configuration by responsibility and separate configuration from live run state. |
| Built-in memory | At least 842 lines plus tool/schema integration. No explicit memory opt-ins appeared in the small downstream constructor sample. Consider a separate integration implemented through ordinary tools; this sample is not proof that nobody uses it. |
| Logging and CLI presentation | About 860 lines across the logger/CLI modules, coupled to runtime calls and global sinks. Consider a small event interface with optional presentation adapters. |
| Public API organization | The dependency-copying/private facades have been removed. Consolidate the remaining agent mixins into coherent components before expanding the public API. |
| Validation/automatic extension loops | These can add implicit model calls. The downstream sample uses final-answer checks and step-extension configuration, so retain their behavior until a replacement execution policy is designed. |

The downstream sample contains nine directly identifiable constructor calls across
three projects; dynamic calls, subclasses, and runtime values are not fully captured.
Feature removal decisions should use observed behavior and product needs, not line
count alone. IkaGeneral is outside the active runtime scope.

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

The latest **8/8** live Codex result, recorded on 2026-10-17, used the installed
2.0.0 wheel at runtime revision `1f2bab0` with `gpt-5.6-sol`:
sync/async text, sync/async tool calls, simple and two-stage execution,
fresh-process resume and cancellation. Resume used no network and preserved usage;
cancellation closed the response while leaving the caller's client usable.
This run includes the historical-replay, workflow HITL, transport cancellation,
nested-frame, transcript, replay-state, stage-budget, fork and SQLite fixes. An earlier
8/8 run at `183ce24` (runtime `344348c`) predates them.

These results cover one model and account. OpenAI API Chat/Responses, Anthropic,
DeepSeek, Gemini, OpenRouter and Z.AI remain unverified live in this review. Existing
offline coverage does not establish live compatibility. Reproduce the Codex run with
[the live-check command](../../CONTRIBUTING.md#live-codex-integration-checks).

## Measured quality and footprint

The full local quality run measured at runtime/test revision `0ec020a` on CPython
3.13.15 passed **872 tests**, with **6 live-provider checks skipped**:
**95.72% line coverage**, **87.19% branch
coverage**, and **93.85% combined coverage**. Ruff, Pyright, compilation, and the isolated
wheel build/import check also passed. The matrix assigns all **121 runtime modules** to
**16 areas** and passed all **102 required scenarios**, with zero gate failures. Coverage
excludes 458 lines, including type-only and abstract declarations. Existing global and
per-area coverage floors are unchanged. At the same revision, the **28 HTTP/HTTPS
real-socket tests passed 20 consecutive rounds (560 cases)**, including deadline timing
and deterministic cancellation after a response hook releases a pooled connection.

The fresh `ikacore-2.0.0-py3-none-any.whl` measured at `0ec020a` is **206.22 KiB**
(211,173 bytes), with **672.89 KiB** of unpacked content (689,041 bytes). The runtime
retains **6 settings** and one direct dependency (`httpx`). The three runtime packages
contain 17,213 physical Python lines, **2,161 more** than `origin/main` at `0f6234c`
(15,052 lines); this count includes blank and comment lines and excludes tests/docs.

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

### Local coverage snapshot (`0ec020a`)

| Area | Modules | Lines | Branches | Floors L/B | Tests observed | Excluded lines |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Public API / package imports | 4 | 98.00% | 90.91% | 95/90% | 377 | 12 |
| Request controls / cancellation | 6 | 95.51% | 90.91% | 95/90% | 298 | 0 |
| HTTP ownership / TLS / connection reuse | 5 | 95.95% | 92.86% | 95/90% | 219 | 2 |
| Retries / request errors | 2 | 87.34% | 77.38% | 85/75% | 175 | 0 |
| Tools / schemas / output bounds | 14 | 95.57% | 88.74% | 95/85% | 294 | 161 |
| Snapshot storage / privacy / replay rules | 8 | 94.67% | 90.32% | 90/90% | 99 | 0 |
| Agent snapshots / human resume | 6 | 98.61% | 93.27% | 95/90% | 120 | 0 |
| Workflow snapshots / selective replay | 3 | 98.65% | 92.19% | 95/85% | 53 | 0 |
| Workflows / fan-in / instance isolation | 9 | 98.25% | 93.24% | 95/90% | 104 | 66 |
| Memory integration | 9 | 93.30% | 77.27% | 90/75% | 19 | 10 |
| Logging / CLI | 3 | 96.76% | 88.46% | 95/85% | 344 | 69 |
| Agent runtime / stages | 15 | 98.17% | 87.41% | 95/80% | 439 | 138 |
| Summarization / compaction | 1 | 97.81% | 84.62% | 95/80% | 34 | 0 |
| Codex auth / streaming / payloads | 7 | 94.70% | 88.94% | 90/80% | 143 | 0 |
| Provider payloads / metadata / history | 19 | 94.25% | 83.74% | 90/80% | 172 | 0 |
| Chat loop / accounting / human interruption | 10 | 95.13% | 84.74% | 90/80% | 80 | 0 |

The generated matrix includes required scenario outcomes, exact uncovered locations,
and known gaps. CI artifacts describe each checked revision.
