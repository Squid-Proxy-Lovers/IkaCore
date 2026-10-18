# Integrated v2 ergonomics / reasoning / hardening validation

Validated runtime/test source revision: **`f8c6c1a2105efc6bb2972b075b4fc4bc51829e1c`**.

Measured **2026-10-18** in `/private/tmp/ikacore-dedupe`, branch `ikacore-v2`,
while resolving the merge of **`7c410f3d90cf7f8721967930caa85fb9754477d8`**
with **`a2b7984ac563f4326c2f20ea4035198df5513f54`**.
This report describes the integrated source with the fixes below, rather than either parent.
Version 2.0.0 remains unreleased. Nothing was pushed or posted.

Runtime Python-content SHA-256:
`72a19681dbe20599ee4d96d945fc697cfa36c6215196a231acc687d975256759`.
Installed app wheel SHA-256:
`afd623a0087f6fd2b86a4b9841a6f3523c3d7b4101f72354a7b7b62070c767fb`.
The separate fresh wheel built by `live_matrix.py` has the same runtime fingerprint.
Its driver records Git HEAD `7c410f3` because the merge was still uncommitted at measurement;
that field is the parent revision, not the integrated runtime identity.

Historical results remain labeled separately:
[pre-ergonomics merge at 8efd658](merge-validation-before-ergonomics.md),
[hardening at dba0e05](hardening-validation.md),
[reasoning branch](REASONING_REPLAY.md), and
[ergonomics-only at ed55da5](ergonomics-handoff.md).

## Final sequential gates

Environment: worktree `.venv`, CPython 3.13.15, HTTPX 0.28.1, Hypothesis 6.170.1.

1. `pytest`: **1,386 passed, 2 skipped**, **31.59 seconds**, no xfails or warnings.
2. `python scripts/quality.py`: **PASS**. Ruff, all-source compilation, Pyright
   (**0 errors / 0 warnings**), the same full suite under coverage, coverage matrix,
   fresh wheel inventory/package data and isolated wheel imports pass.
3. `python scripts/execution_parity.py --output artifacts/execution-parity-integrated.json`:
   **42 traces** generated. The permanent tests compare model-call sequences, budgets
   and snapshot boundaries against `dc9efbd`, asserting the new validation status before
   comparing the historical trace.
4. `python scripts/live_replay_repro.py`: **seven provider reasoning replay probes pass**.
5. A fresh wheel was built after parity, installed non-editably into
   `artifacts/installed-runtime`, and its Python files compared byte-for-byte with source.
   All six apps passed offline; the installed-wheel subprocess suite plus cancellation
   reports **7 passed in 2.01 seconds**.
6. All six apps ran sequentially with `--live --use-local-login --model gpt-5.6-sol`:
   **6/6 passed, 39 request boundaries, 101.76 seconds**.
7. `python scripts/live_matrix.py --provider codex --use-local-login --model gpt-5.6-sol`:
   **14 passed / 0 failed / 1 skipped**, 60-second per-case limits, one live request attempt.
8. The opt-in **2,200-completed-turn soak** was rerun on the final runtime and passed,
   **22.10 seconds** (details below).

The normal skips are the opt-in live smoke entry point and sustained soak, both separate
from the explicit installed-wheel/live/soak runs above. The ordinary suite includes all
**46 process-kill/concurrent-writer cases**, generative tests and controlled summary tests.
The three original DOGFOOD reproductions pass as ordinary assertions. No xfail remains.

Socket-dependent tests require sandbox escalation for loopback HTTP/HTTPS servers.
An initial restricted run failed only at those 59 bind operations; final gates used socket
access. No tests, replay-safety checks or coverage floors were disabled.

## Coverage

- **100.0000% lines**: **9,772 / 9,772 statements**.
- **97.1816% branches**: **2,655 / 2,732 arcs**;
  **77** uncovered arcs.
- **99.3842% combined** statement/branch coverage.
- **129 runtime modules / 16 areas**, **149 required area/scenario checks**
  (**146 unique selectors**), **zero gate failures**.
- Global floors remain **99% lines / 95% branches / 99% combined**.
  Every per-area hardening floor was compared with `7c410f3` and preserved exactly.
- **201 excluded lines**: the prior 194 plus seven automatically excluded type-only
  imports/guards introduced by ergonomics. No `no cover` or `no branch` pragma was added.

| Area | Modules | Lines | Branches | Floors L/B |
| --- | ---: | ---: | ---: | ---: |
| Public API / package imports | 4 | 100.00% | 95.00% | 99/95% |
| Request controls / cancellation | 6 | 100.00% | 95.83% | 99/95% |
| HTTP ownership / TLS / connection reuse | 5 | 100.00% | 98.75% | 99/98% |
| Retries / request errors | 4 | 100.00% | 99.09% | 99/98% |
| Tools / schemas / output bounds | 11 | 100.00% | 97.34% | 99/97% |
| Snapshot storage / privacy / replay rules | 8 | 100.00% | 100.00% | 99/99% |
| Agent snapshots / human resume | 6 | 100.00% | 95.37% | 99/95% |
| Workflow snapshots / selective replay | 3 | 100.00% | 97.14% | 99/96% |
| Workflows / fan-in / instance isolation | 9 | 100.00% | 98.30% | 99/97% |
| Memory integration | 10 | 100.00% | 99.17% | 99/99% |
| Logging / CLI | 5 | 100.00% | 93.75% | 99/93% |
| Agent runtime / stages | 18 | 100.00% | 93.84% | 99/92% |
| Summarization / compaction | 2 | 100.00% | 97.22% | 99/96% |
| Codex auth / streaming / payloads | 7 | 100.00% | 98.68% | 99/98% |
| Provider payloads / metadata / history | 20 | 100.00% | 97.99% | 99/97% |
| Chat loop / accounting / human interruption | 11 | 100.00% | 96.57% | 99/96% |

Exact branch locations and test contexts remain in `artifacts/quality/coverage.json`
and `coverage-matrix.json`. The old [hardening branch inventory](hardening-branches.md)
remains historical at `dba0e05`; its arcs are not relabeled as current.
Coverage demonstrates exercised paths, not universal correctness.

## Conflict resolutions

- **`scripts/coverage_manifest.json`:** retain both the hardening retry regression and
  the new mock-provider scenario. Inventory all four new runtime modules and keep every
  raised floor; register additional integration scenarios.
- **`agent_snapshots.py`:** load saved conversation, prompt, counters and retry state
  before returning completed output. Restore usage/cost and atomically finalize run/root
  frame metadata after a completion-boundary crash, without repeating model/tool effects.
  Reject `resume_input` for a completed snapshot; pending snapshots retain replay checks.
- **`chat_request.py`:** retain hardening's successful-summary-only live-buffer trimming
  and live messages at pre/post snapshot boundaries, while invoking ergonomics' successful
  compaction callback/event exactly once. Preserve cancellation and typed malformed-response
  handling in both sync/async proactive and overflow recovery paths.

## Semantic integration fixes and checks

1. **Conversation compaction:** structural retention previously recognized stage/HITL
   inputs but omitted ergonomics' `user_input`. It now retains the latest `send()`
   instruction alongside complete/pending tool exchanges and consecutive native reasoning.
2. **Token budget estimation:** `first_input` can mirror a recorded opening user turn.
   Count that input once, matching provider replay, while counting distinct instructions
   independently. Malformed/unhashable history entry types preserve the defensive estimate.
3. **Structured synchronous context errors:** synchronous `compress_hook` failures now
   raise `WorkflowError` with node identity and completed partial results, matching parallel
   preparation. Cancellation/deadline exceptions retain their original types. The explicit
   exception-boundary inventory records this application-callback conversion.
4. **Cancellation and worker cleanup:** regressions exercise worker-raised cancellation,
   outstanding parallel work, and exception propagation through executor completion.
   Dependent work remains blocked after errors.
5. **Borrowed clients versus deadline pools:** high-level workflow tests pass injected
   client headers, cookies and auth through clones and controlled requests; private request
   cleanup and clone shutdown preserve the caller client. Late custom work after a deadline
   is released before the test ends. Owned async clients close separately.
6. **Reasoning across multi-turn restore:** all seven public SQLite provider fixtures now
   restore a completed conversation and call `send()` again, verifying original opaque
   state, exactly one new instruction, and one new usage charge. Existing malformed native
   parsing and consecutive-reasoning compaction regressions remain intact.
7. **Output scopes:** compaction events retain workflow instance IDs and stay in each
   agent's event scope. Clone counters do not mutate the template; diagnostic loggers
   remain separate from scoped runtime output.
8. **Soak recovery and defaults:** sustained fixtures use explicit large budgets and
   durability workflows retain explicit worker limits. The soak now selects the latest
   exact root-frame checkpoint, rather than relying on incidental compaction timing or
   implicitly replaying a mid-tool snapshot. Querying one frame also keeps fixture
   inspection from materializing the entire append-only audit log during memory measurement.
9. **Canonical APIs:** tests, durability workers, benchmarks, parity/live scripts and examples
   use `stages=` and workflow `parallel=`. Explicit tests retain the compatibility aliases.
   Workflow output/transcript/instance maps and fork-with-answer survive cached replay.
   Validation correction and fail-closed behavior run without replaying application tools.
10. **Fixture contracts:** summary mocks return actual strings, avoiding coroutine warnings
    from unspecified AsyncMock values. The original dogfood assertions are retained without
    their three xfail markers.

## Sustained run on the integrated runtime

- **2,200 completed turns**, **2,201 provider attempts /
  replayable read-tool calls**, **733 compactions**, **6 cancellations**.
- Mid-loop cancellation resumes a safe exact snapshot. Completed replay adds no request,
  effect or usage. Final accounting is **2,200 input + 2,200 output = 4,400 tokens**.
- **14,678 snapshots**, **75,026,432 database bytes**;
  SQLite integrity passes and disk growth remains within the per-turn/slope assertions.
- Retained memory change after the third sample: **21,739 bytes**; traced peak
  **15,595,554 bytes**. Threads **2 → 2**;
  file descriptors **12 → 12**.
- 220-turn batches range **2.139–2.266 seconds**,
  within the late/early latency bound.

This single mocked workload does not establish indefinite leak freedom or power-loss
durability. The historical live fact-retention evaluation remains **19/20**, rather than
being promoted to a perfect-retention claim.

## Installed-wheel Codex matrix

Runtime fingerprint matches the six-app wheel. The reasoning case skipped because the
backend exposed no reasoning item; this live run does not establish native reasoning replay.
Seven-route signed/encrypted replay is verified offline, and older high-effort live evidence
remains explicitly historical in [REASONING_REPLAY.md](REASONING_REPLAY.md).

| Case | Result | Reason | Seconds |
| --- | --- | --- | ---: |
| sync_text | passed |  | 1.86 |
| async_text | passed |  | 1.56 |
| sync_tools | passed |  | 4.87 |
| async_tools | passed |  | 3.68 |
| simple_agent | passed |  | 1.60 |
| staged_agent | passed |  | 3.59 |
| fresh_process_resume | passed |  | 1.87 |
| async_cancellation | passed |  | 1.17 |
| reasoning_tools | skipped | reasoning_not_exposed | 2.88 |
| streaming_sse | passed |  | 1.44 |
| usage_accounting | passed |  | 1.70 |
| deadline | passed |  | 0.15 |
| transient_retry | passed |  | 0.15 |
| compaction | passed |  | 6.70 |
| workflow_hitl | passed |  | 5.51 |

The synthetic transient-retry probe uses a fixture transport; expired deadlines make no
model request. Cancellation and skipped reasoning have incomplete observable usage, so
reported token totals are not billing records. Only Codex was exercised live.

## Documentation and retained evidence

One coherent CHANGELOG 2.0.0 entry, the migration table, README/USAGE, generated API reference,
dogfood/handoff history and [application results](../../examples/apps/LIVE_RESULTS.md) now
describe the same public contracts. API completeness is checked by the full pytest suite.
Historical measurements retain their own revisions rather than being relabeled.

Ignored local evidence:
`artifacts/{pytest-final.log,quality-final.log,sequential-checks.json,execution-parity-integrated.json,reasoning-replay-final.log,merge-soak.log,merge-soak.json}`,
`artifacts/quality/{coverage.json,coverage-matrix.json,coverage-matrix.md,tests.xml,*.whl}`,
`artifacts/integrated-apps-{offline,live}/`,
`artifacts/installed-wheel/` and `artifacts/integrated-live-matrix/`.
The committed sanitized app JSON retains all attempts and earlier histories.
