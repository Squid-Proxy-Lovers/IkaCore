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

These are source/maintenance reductions. Runtime latency and allocation changes have
not been benchmarked. Tests, static typing, package builds, and provider/workflow
regressions remain required.

## Breaking changes and migration

| Previous call or behavior | V2 behavior / migration |
| --- | --- |
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

The remaining runtime settings are `shared_ssl_context`, `reuse_connections`,
`parallel_context_preparation`, `recover_truncated_control_calls`, `compact_at_fraction`,
and `summary_context_threshold`. These configure resource ownership, concurrency, and
context policy rather than switching between v1 and v2 implementations.

## Retained snapshots

```python
from IkaCore import RuntimeControl

store = agent.enable_snapshots("runtime-data/snapshots.db", RuntimeControl(pause_points={"post_model"}))
paused = agent.execution()
result = agent.resume_from_snapshot(paused["checkpoint_uid"])
```

Agent snapshots record run/stage/tool/subagent frames, history, usage, and events.
Post-model resume consumes the saved response without counting usage again; completed
snapshots return saved output. Fork/restart creates a new run. Staged human-input resume
accepts `resume_input`. Mid-tool replay restarts execution and requires an explicit
unsafe-replay decision; it cannot restore a Python stack.

Workflow snapshots are enabled with `workflow.enable_snapshots(path, control)` and
started with `workflow.run(...)`. `snapshot_runtime_info()` exposes the last snapshot.
Completed instances may be reused; selective replay invalidates downstream results.
Nested and unfinished tool effects are checked before replay. Snapshot JSON filters
known credential patterns and redacted state is refused for replay by default. Filtering
cannot identify all private content, so keep runtime databases private.

## Remaining candidates for simplification

| Area | Evidence and proposed direction |
| --- | --- |
| Agent configuration/inheritance | `IkaBaseAgent` accepts 37 constructor arguments; the runtime has many small mixins and duplicated simple/staged loops. Group configuration by responsibility and consolidate the execution state machine in a separate focused change. |
| Checkpoints plus snapshots/replay | Snapshot-related modules account for about 1,575 lines, alongside the old checkpoint API. Choose one persistence model; establish which fork/selective-replay use cases belong in core before expanding it. |
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

## Measured quality and footprint

The quality runner now produces an `artifacts/quality/coverage-matrix.md` report
for all 122 runtime modules across 16 subsystems. It tracks 53 required scenarios plus
missing branches and known gaps. Current local results: **95.32% line coverage, 85.32%
branch coverage**, with 501 lines excluded by coverage's rules (including type-only and
abstract declarations). Live-provider behavior and model/summary quality remain unverified
by the offline suite. Request-control, transport and snapshot areas now have independent
90% branch floors.

An isolated runtime-only wheel installation on macOS arm64 / CPython 3.11.16 measured:

| Measurement | Result |
| --- | ---: |
| Wheel download | 205.1 KiB |
| Unpacked wheel content | 675.3 KiB |
| Installed distributions including dependencies and bytecode | 5.40 MiB |
| Direct runtime dependency | 1 (`httpx`); 7 dependency distributions total |
| Public package imports, fresh interpreter | 3.47 ms |
| Full agent runtime import, fresh interpreter | 32.98 ms |
| Peak RSS after runtime import, including Python/worker | 32.59 MiB |
| Python/worker baseline peak RSS | 19.23 MiB |
| HTTPX-only import peak RSS | 27.16 MiB |
| Agent construction and shutdown, warmed imports | 0.0043 ms |
| Mock HTTPX request/client lifecycle | 0.0461 ms |
| One-step synchronous agent, mocked provider | 0.231 ms |
| One-step async agent via sync bridge, mocked provider | 0.657 ms |
| Same agent with SQLite snapshots | 11.81 ms |
| Single 4 KiB snapshot append | 0.586 ms |
| Pure four-node workflow, sequential / parallel | 0.008 / 0.247 ms |

Timings are medians of nine fresh-process samples, 100 warm operations each. They exclude
real provider/model time; full agent measurements include construction/client/shutdown.
Imports use fresh interpreters but a warmed OS filesystem cache. These results describe
this machine and fixture, not production latency guarantees. Reports separate batch-average
P95, interpreter wall time, and RSS. The first 1,000 repeated agent runs retained about
1.08 MiB of traced allocations; the next 1,000 reduced retained allocations by about 46 KiB.
That short run is consistent with cache warmup, not proof of leak freedom under every workload.

The default package and basic execution path are small. Persistence adds substantial local
I/O cost, parallel scheduling costs more for trivial work, and the 17,000+ line runtime still
has architectural complexity worth reducing. These numbers support evaluating optional
features separately. See [benchmark instructions](../../CONTRIBUTING.md#reproducing-the-offline-benchmarks)
and the generated `artifacts/benchmarks/benchmark.json` report. CI publishes both reports as run artifacts.

### Local coverage snapshot

| Area | Modules | Lines | Branches | Floors L/B | Tests observed | Excluded lines |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Public API / package imports | 4 | 95.97% | 90.91% | 95/90% | 298 | 12 |
| Request controls / cancellation | 6 | 95.53% | 90.91% | 95/90% | 154 | 0 |
| HTTP ownership / TLS / connection reuse | 5 | 95.14% | 91.30% | 95/90% | 105 | 2 |
| Retries / request errors | 2 | 87.42% | 77.91% | 85/75% | 131 | 0 |
| Tools / schemas / output bounds | 14 | 95.54% | 88.67% | 95/85% | 204 | 161 |
| Snapshot storage / privacy / replay rules | 8 | 94.98% | 91.82% | 90/90% | 35 | 0 |
| Agent snapshots / human resume | 6 | 98.08% | 91.49% | 95/90% | 29 | 0 |
| Workflow snapshots / selective replay | 3 | 96.82% | 88.89% | 95/85% | 24 | 0 |
| Workflows / fan-in / instance isolation | 9 | 98.25% | 93.24% | 95/90% | 40 | 66 |
| Memory integration | 9 | 93.23% | 76.14% | 90/75% | 17 | 10 |
| Logging / CLI | 3 | 96.58% | 86.92% | 95/85% | 218 | 69 |
| Agent runtime / stages / checkpoints | 17 | 97.63% | 82.43% | 95/80% | 370 | 181 |
| Summarization / compaction | 1 | 97.81% | 84.62% | 95/80% | 31 | 0 |
| Codex auth / streaming / payloads | 6 | 93.51% | 84.58% | 90/80% | 120 | 0 |
| Provider payloads / metadata / history | 19 | 93.85% | 82.30% | 90/80% | 163 | 0 |
| Chat loop / accounting / human interruption | 10 | 94.04% | 81.58% | 90/80% | 71 | 0 |

The generated matrix lists all 53 required scenario IDs, their JUnit outcomes, exact uncovered
locations, and per-area gaps. The manifest and generator are versioned; CI artifacts reflect
each checked revision.
