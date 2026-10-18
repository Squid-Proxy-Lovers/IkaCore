# Reasoning / dogfood / hardening merge validation

Measured on 2026-10-18 at runtime/test revision **`8efd65803e5d2a55c8039ee67ce18c4f9c4e5464`**,
following merge `4dda845` on `ikacore-v2`. Work was confined to
`/private/tmp/ikacore-dedupe`; agent APIs were left for the concurrent ergonomics work.
Version 2.0.0 remains unreleased. Nothing was pushed or posted.

## Sequential final checks

Using the worktree `.venv`, CPython 3.13.15, HTTPX 0.28.1 and Hypothesis 6.170.1:

1. `pytest`: **1,320 passed, 2 skipped, 3 xfailed**, 31.18 seconds.
2. `python scripts/quality.py`: **PASS**. Its coverage run also reports **1,320 passed,
   2 skipped, 3 xfailed**, 44.23 seconds. Ruff, compilation, Pyright (**0 errors /
   0 warnings**), coverage matrix, fresh wheel inventory/package data and isolated
   installed-wheel imports pass.

The skips are the opt-in installed-wheel live smoke entry point and sustained soak.
The three strict expected failures are the existing DOGFOOD-1/2/3 final-validation
and prompt-update findings in [dogfood-report.md](dogfood-report.md). This integration
keeps those failures visible; it does not claim to resolve their agent API contracts.
The normal full suite includes all **46 process-kill/concurrent-writer cases**.
No additional live provider requests or opt-in soak were run for this revision.

The first restricted coverage run failed 59 socket-dependent cases with
`PermissionError: [Errno 1] Operation not permitted` at loopback server bind.
The final sequential runs used sandbox escalation for those local HTTP/HTTPS tests.
No tests or coverage floors were disabled.

## Measured coverage

The pre-reconciliation coverage artifact at merge `4dda845` reported **99.6931% lines**,
**96.0437% branches**, and **98.8929% combined**. The last number failed the 99%
pytest-cov gate; it was not the separate line percentage.

At `8efd658`:

- **100.0000% lines**: **9,439 / 9,439 statements**, no uncovered runtime lines.
- **96.8608% branches**: **2,561 / 2,644 arcs**, 83 uncovered arcs.
- **99.3131% combined** statement/branch coverage.
- **194 pre-existing excluded lines**, unchanged; no new `no cover` or `no branch`
  pragmas were added.
- Global floors remain **99% lines / 95% branches / 99% combined**. Every raised
  hardening area floor remains intact.
- **125 runtime modules**, **16 areas**, **141 required area/scenario checks**
  (139 unique selectors), **zero gate failures**. The new diagnostics module and
  reasoning merge regressions are registered in the manifest.

| Area | Modules | Lines | Branches | Floors L/B |
| --- | ---: | ---: | ---: | ---: |
| Public API / package imports | 4 | 100.00% | 95.00% | 99/95% |
| Request controls / cancellation | 6 | 100.00% | 95.83% | 99/95% |
| HTTP ownership / TLS / connection reuse | 5 | 100.00% | 98.75% | 99/98% |
| Retries / request errors | 2 | 100.00% | 98.84% | 99/98% |
| Tools / schemas / output bounds | 11 | 100.00% | 97.34% | 99/97% |
| Snapshot storage / privacy / replay rules | 8 | 100.00% | 100.00% | 99/99% |
| Agent snapshots / human resume | 6 | 100.00% | 95.28% | 99/95% |
| Workflow snapshots / selective replay | 3 | 100.00% | 96.88% | 99/96% |
| Workflows / fan-in / instance isolation | 9 | 100.00% | 98.00% | 99/97% |
| Memory integration | 10 | 100.00% | 99.17% | 99/99% |
| Logging / CLI | 5 | 100.00% | 93.75% | 99/93% |
| Agent runtime / stages | 16 | 100.00% | 92.48% | 99/92% |
| Summarization / compaction | 2 | 100.00% | 96.30% | 99/96% |
| Codex auth / streaming / payloads | 7 | 100.00% | 98.68% | 99/98% |
| Provider payloads / metadata / history | 20 | 100.00% | 97.44% | 99/97% |
| Chat loop / accounting / human interruption | 11 | 100.00% | 96.50% | 99/96% |

Exact remaining line/branch locations and test contexts are retained in
`artifacts/quality/coverage.json` and `coverage-matrix.json`. The older
[hardening branch inventory](hardening-branches.md) is explicitly historical at
`dba0e05`; its 81 arcs are not relabeled as this revision's 83 arcs.
Coverage demonstrates exercised paths, not universal correctness.

## Fixes and semantic merge checks

- **Defensive provider test port.** The removed OpenAI `_message_content` helper
  was replaced by the actual message normalizer. Scalar wrapping, invalid history,
  developer-role handling and required-tool selection assertions all run again.
  Responses scalar wrapping is tested through the current item converter.
- **Cohesive module split.** Opt-in request/response dump creation and its best-effort
  filesystem handling now live in `chat_interface/debug_dump.py`. Request code and
  tests call that owner directly; no private forwarding facade was introduced.
  `chat_response.py` drops from **604 to 561 physical lines**, below the unchanged
  600-line limit. The diagnostics module is 60 lines and has strict typing enabled.
- **Dead context builders removed.** Five unused Responses/Codex scalar/context and
  first-message helpers were superseded by `ordered_replay_messages`. No coverage
  exclusions or floor changes were used to accommodate them.
- **Semantic conflict: malformed native replay.** The display parsers could ignore
  scalar items while replay capture stored the entire mixed collection next to valid
  reasoning. Finalization then called `.get()` on those scalars, or the next payload
  replayed malformed blocks. Replay capture now validates native collections and
  objects before storing pending state; the existing parse boundary raises
  `IkaProviderResponseError`. Twenty-eight new cases cover scalar collections and
  scalar/null/list elements for Responses, Codex, Anthropic and Gemini, assert input
  immutability, and assert that no invalid pending state is persisted.
- **Semantic conflict: consecutive native reasoning items.** Compaction reset the
  native group start on every reasoning item and dropped earlier items before a
  tool call. It now preserves the start of the whole contiguous reasoning/call group.
  The failing-before regression checks both encrypted items and the matching result.
- **Compaction across the new persisted envelopes.** Fourteen new cases exercise all
  seven provider routes with both a live buffer and a fresh buffer. They parse signed /
  encrypted reasoning, append its tool exchange, compact persisted and live history,
  and rebuild a provider payload. Assertions retain original exchange objects and
  opaque values, preserve call/result pairing exactly once, and remove old noise.
- **Snapshot accounting remains correct with reasoning history.** Seven public SQLite
  regressions now use nonzero provider usage. Fresh-instance HITL resume preserves
  opaque history and reaches exactly **8 input / 4 output tokens** after two requests.
  A third fresh instance reopens the completed snapshot with identical usage/cost
  and reasoning history, without a third provider request. Existing crash-recovery
  tests also pass unchanged.
- **Cancellation remains a control error.** Both sync and async forced-completion
  regressions now carry pending encrypted reasoning. Cancellation propagates unchanged
  and leaves the entire history, including pending replay state, unmodified.

The semantic regressions initially produced **17 failures / 14 passes** before the
production fixes. The final added regression module has **43 passing cases**.
No agent configuration, constructor, prompt or validation policy implementation changed.

## Documentation and historical evidence

[README.md](README.md) now distinguishes current integration measurements from the
`179b52e` refactor, `01249f6` reasoning and `dba0e05` hardening results. The changelog
consolidates provider replay entries and includes hardening/application work without
claiming unchanged coverage floors. Historical live reasoning and compaction evidence
retains its original scope: the bounded Codex high-effort replay probe passed; the
four hardening compaction fixtures scored **19/20** strict follow-up facts. Neither
result is presented as a live rerun of `8efd658`.

## Reproduction and artifacts

Run sequentially from `/private/tmp/ikacore-dedupe`, with loopback access:

```bash
export PATH="$PWD/.venv/bin:$PATH" VIRTUAL_ENV="$PWD/.venv"
pytest
python scripts/quality.py
```

Ignored evidence retained locally:

- `artifacts/quality/pytest-merge-final.log`, `quality-merge-final.log`, `tests.xml`.
- `artifacts/quality/coverage.json`, `coverage-matrix.json`, `coverage-matrix.md`.
- `artifacts/quality/ikacore-2.0.0-py3-none-any.whl`: **224,901 bytes**, SHA-256
  `7c6d2fda4aae397fea5ea6306db3e20b6ec7e808a314b5974dd008948dccf9d3`.
- `/private/tmp/ikacore-reasoning-before.log`: failing-before semantic regressions.
- `/private/tmp/ikacore-dedupe-baseline.log`: restricted coverage run and socket errors.

## Local commits

- `8efd65803e5d2a55c8039ee67ce18c4f9c4e5464`: targeted runtime/test/manifest reconciliation.
- The documentation-only commit containing this report follows that revision; it does
  not change the measured runtime or tests.

Each commit ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
