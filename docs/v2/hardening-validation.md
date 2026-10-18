# v2 hardening validation

Work was confined to `/private/tmp/ikacore-w-hardening`, branch `v2-hardening`, starting at `aca8c53a57332c0e5ab225628ac7b29bae99d23b`. Version 2.0.0 remains unreleased. Nothing was pushed or posted. Provider payload/history modules and example apps were not edited.

## Final offline gates

- `pytest` inside `python scripts/quality.py`: **1,211 passed, 2 skipped**. The skips are the installed-wheel live smoke entry point and the explicitly opt-in soak.
- Ruff, compilation, Pyright (**0 errors / 0 warnings**), coverage manifest, fresh wheel module inventory and isolated wheel imports: **passed**.
- Line coverage: **96.3850% → 100.0000%** (8,852/9,184 → 9,305/9,305 statements).
- Branch coverage: **88.1955% → 96.8629%** (2,219/2,516 → 2,501/2,582 arcs).
- Combined coverage: **94.6239% → 99.3186%**.
- Global floors: **94/85% → 99/95% lines/branches**; the combined pytest-cov floor rises **91% → 99%**. All 16 area floors also rise.
- Inventory: **122 → 123 runtime modules**, including the new structural-compaction owner; **133 required area/scenario checks**, all passing.
- No uncovered runtime lines remain. **194 pre-existing exclusions remain; zero exclusions were added.** The [complete remaining branch inventory](hardening-branches.md) lists all 81 uncovered arcs and their conditions. Coverage is evidence of exercised paths, not proof that every input works.

Measured with the isolated `.venv`, CPython 3.13.15, HTTPX 0.28.1 and Hypothesis 6.170.1. Heavy runs were sequential. Loopback transport tests required execution outside the socket-restricted sandbox; the first restricted baseline failed for that environment reason, and the authorized baseline passed.

## Before/after coverage by area

| Area | Lines before → after | Branches before → after | Raised floors L/B |
| --- | ---: | ---: | ---: |
| Public API / package imports | 97.89 → 100.00% | 90.00 → 95.00% | 99/95% |
| Request controls / cancellation | 99.63 → 100.00% | 93.75 → 95.83% | 99/95% |
| HTTP ownership / TLS / connection reuse | 97.76 → 100.00% | 95.00 → 98.75% | 99/98% |
| Retries / request errors | 87.66 → 100.00% | 78.57 → 98.81% | 99/98% |
| Tools / schemas / output bounds | 96.13 → 100.00% | 90.83 → 97.34% | 99/97% |
| Snapshot storage / privacy / replay rules | 94.89 → 100.00% | 91.13 → 100.00% | 99/99% |
| Agent snapshots / human resume | 99.46 → 100.00% | 94.23 → 95.28% | 99/95% |
| Workflow snapshots / selective replay | 98.65 → 100.00% | 92.19 → 96.88% | 99/96% |
| Workflows / fan-in / instance isolation | 98.28 → 100.00% | 93.33 → 98.00% | 99/97% |
| Memory integration | 93.75 → 100.00% | 76.67 → 99.17% | 99/99% |
| Logging / CLI | 97.86 → 100.00% | 88.75 → 93.75% | 99/93% |
| Agent runtime / stages | 98.18 → 100.00% | 87.22 → 92.48% | 99/92% |
| Summarization / compaction | 97.81 → 100.00% | 86.54 → 96.30% | 99/96% |
| Codex auth / streaming / payloads | 95.69 → 100.00% | 90.87 → 98.26% | 99/98% |
| Provider payloads / metadata / history | 94.75 → 100.00% | 84.98 → 97.74% | 99/97% |
| Chat loop / accounting / human interruption | 95.99 → 100.00% | 86.32 → 96.46% | 99/96% |

## Reproduced bugs and fixes

1. **Completion recovery lost accounting and left metadata running.** Kill the process immediately after the committed `agent_completed` boundary. A fresh agent previously returned saved output with zero live usage counters and an unfinished run. Completed-snapshot replay now restores history, usage and cost, and atomically finalizes frame/run metadata without another model/tool call. Reproduction/regression: `test_killed_agent_recovers_only_committed_boundaries` with `after:agent_completed`, both exit methods and journal modes.
2. **Malformed provider JSON leaked an internal exception.** Hypothesis shrank a failure to `{"choices": "0"}`, which leaked `AttributeError` through `_parse_provider_round`. The shared parse boundary now wraps that failure in `IkaProviderResponseError`; provider payload modules were untouched. Regression: `test_nested_malformed_provider_shapes_never_escape_typed_parse_errors`.
3. **Forced completion swallowed cancellation.** A summarizer raising `IkaRequestCancelled` became a successful `AgentEndException` in both sync and async finalization. Both regressions failed before the fix (`cancel-before.log`); cancellation now propagates with unchanged history. Regression: `test_force_completion_never_converts_cancellation_to_success`.
4. **Compaction omitted constraints/reasoning and discarded required structure.** Summary input now includes system constraints and separately stored reasoning. Successful compaction retains the latest user instruction, the latest complete tool exchange, unresolved calls and associated reasoning. Chat, Anthropic and Gemini tool references and native parallel Responses items are retained without orphaning results or dropping encrypted reasoning. An empty/whitespace summary no longer destroys history. Regressions: `test_controlled_compaction_contains_facts_and_keeps_pending_structure`, `test_empty_summary_does_not_erase_history`, and native/provider structural fixtures.
5. **Compaction left the live provider buffer growing.** A 1,005-message sync/async fixture still sent its old turns after summarizing saved history. Proactive and context-overflow compaction now trim that buffer before the next payload is built; pre/post snapshot boundaries capture the corresponding buffer. Regressions: `test_compaction_also_bounds_the_live_request_buffer` and `test_context_overflow_retry_uses_compacted_live_buffer`.

The structural logic lives in `IkaModel/compaction.py` to preserve the existing production-module size limit. No gate or lint rule was relaxed.

## Crash durability and concurrent writers

**46 cases passed**, using real child interpreters and fresh-process recovery. Agent kill points cover an uncommitted pre-model snapshot, post-model snapshot/event inserts, committed pre/post-model boundaries, mid-tool execution, an uncommitted completion snapshot and a committed completion boundary. Each agent point runs with both `os._exit(73)` and SIGKILL, in DELETE and WAL modes.

Recovery asserts `PRAGMA integrity_check = ok`, complete snapshot/event pairs, committed-only boundary visibility, completion/accounting restoration, and exactly one externally recorded effect. Continuations that could repeat a completed/partial `replay_policy="deny"` effect are refused before calling the mock provider or writing another effect. Safe saved post-model responses resume without another request or usage charge.

Parallel-workflow kills cover workflow entry, completed parallel-instance boundaries and workflow completion. Fresh processes preserve committed siblings and complete remaining work; additional cases use real non-idempotent tool writes and either reuse committed results or deny unsafe continuation. Two simultaneous processes each append 30 events to one database: all 60 sequence numbers are unique and contiguous, with no foreign-key violation. Both journal modes retain SQLite FULL synchronous defaults.

This qualifies process termination, not power loss, storage-controller guarantees, distributed scheduling or arbitrary external transactions.

## Sustained offline soak

`IKACORE_RUN_SOAK=1` ran the opt-in test once more against the final runtime before packaging:

- **2,200 completed turns**, **2,201 provider attempts / read-tool calls**, **440 compactions**, **6 cancellations**. A mid-loop cancellation resumes its SQLite snapshot; one interrupted replayable read may repeat. Completed snapshot replay makes no extra provider/tool call.
- **20.48 seconds**, **14,092 snapshots**; database **59,604,992 bytes** with roughly linear append-only growth. Indefinite audit retention is not claimed to have bounded disk size.
- Traced retained-memory change after the third 220-turn sample: **13,569 bytes**; traced peak **15,176,096 bytes** (including fresh-state replay).
- Threads **2 → 2**; file descriptors **12 → 12**. The mock provider opens no network sockets.
- 220-turn batches **1.987–2.203 seconds**, with no late-batch latency escalation.
- Exactly **2,200 input + 2,200 output = 4,400 total** accounted tokens after resume. Integrity check passes; final completed/cancelled runs match expectations.

Assertions enforce memory/resource bounds, disk bytes per turn and growth slope, and a late/early latency bound. This single mocked workload does not prove indefinite leak freedom.

## Controlled and live compaction evaluation

Four long histories include task goals, system prohibitions/numeric budgets, an archived tool result, an older open approval question, the latest user instruction, signed/opaque reasoning and pending calls. The archived result/question are deliberately absent from retained raw entries: the summary is required to carry them. Scripted sync/async summarizers assert the complete input payload and exact summary placement, and preserve system/first-input records, pairing and pending items. Provider-specific live-buffer fixtures include native parallel calls and encrypted reasoning.

The opt-in installed-wheel Codex harness used `--use-local-login --model gpt-5.6-sol --fixtures 4`, one summary and one follow-up per fixture, one live attempt per request and a 60-second fixture budget. It copied the existing login to a private temporary file, then removed the copy. Reports contain numeric usage and fact booleans, not credentials or raw model output.

**Strict follow-up fact retention: 19/20 (95%)**; per fixture **5/5, 4/5, 5/5, 5/5**. Archived tool results and older approval questions scored **4/4 each**. Fixture 1 missed the constraint scorer; that failure is retained, and the evaluator exits nonzero when any fact misses. This is not a perfect-retention claim. The preservation prompt was strengthened to prioritize exact goals, prohibitions, numbers, identifiers, verified results and pending work.

The live run used an interim installed wheel with the same fact fixtures and summary prompt, runtime fingerprint `b8b2a7dba01440966c9355a53115d5f95bcfbe31b5c59eb08412c4810305a7a7`. Subsequent structural-buffer/native-item refinements are covered offline; the report does not relabel that earlier fingerprint as the final wheel.

## Generative testing

Hypothesis is a **test extra only**. Deterministic bounded strategies cover Unicode SSE at arbitrary byte boundaries (including malformed events and CR/LF variants), malformed/truncated provider JSON, nested response shapes for all seven parser routes, privacy-aware nested snapshot round trips and append-only lineage, and tool JSON decoding/argument validation with input immutability. Properties compare sync/async stream outcomes, exact decoded text, typed failures, structural round trips and behavioral results; they do not merely assert execution.

## Reproduction and retained evidence

All commands run from `/private/tmp/ikacore-w-hardening`:

```bash
export PATH="$PWD/.venv/bin:$PATH" VIRTUAL_ENV="$PWD/.venv"
pytest -q
python scripts/quality.py
pytest -q src/IkaTest/test_crash_durability.py
IKACORE_RUN_SOAK=1 IKACORE_SOAK_REPORT="$PWD/artifacts/hardening/soak.json" \
  pytest -q src/IkaTest/test_runtime_soak.py
python scripts/live_matrix.py --retention-eval --use-local-login \
  --model gpt-5.6-sol --fixtures 4 --output-dir artifacts/hardening/live-compaction
```

Retained ignored evidence: `artifacts/hardening/{baseline.log,baseline-compact.json,pytest-final.log,quality-final.log,crash.log,soak.log,soak.json,cancel-before.log,live-buffer-before.log,live-compaction.log,live-compaction/retention.json}` and `artifacts/quality/{coverage.json,coverage-matrix.json,coverage-matrix.md,tests.xml,*.whl}`. The live evaluator intentionally reports a failure at 19/20; offline pytest and the quality gate pass.

## Local commits

- `de36c3a0ffef44f7b6223ed77ac7aa69819377d2`: subprocess crash durability and completion/accounting recovery.
- `e488f97ee331e063270f96bc14b0965d8606c1f2`: compaction, failure propagation, generative tests, soak and live evaluation.
- `e22fdcfc3f2ec004cc7dd42cb552afedb479ab60`: request/provider/memory/presentation edge contracts.
- The documentation/coverage-floor commit containing this report follows those area commits. Every commit ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
