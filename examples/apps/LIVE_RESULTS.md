# Codex application usage results

Measured **2026-10-18T06:43:01.531513+00:00** on the integrated ergonomics/reasoning/hardening source
(merge parents `7c410f3` and `a2b7984`, with integration fixes).
Model: **gpt-5.6-sol**, explicitly using `--live --use-local-login --model gpt-5.6-sol`.
Runtime: fresh non-editable IkaCore 2.0.0 wheel, CPython 3.13.15 / HTTPX 0.28.1.
Imports were verified under `artifacts/installed-runtime/.../site-packages`, without
editable imports or PYTHONPATH.

Wheel SHA-256: `afd623a0087f6fd2b86a4b9841a6f3523c3d7b4101f72354a7b7b62070c767fb`.
Runtime Python-content SHA-256: `72a19681dbe20599ee4d96d945fc697cfa36c6215196a231acc687d975256759`.
[Sanitized JSON evidence](live-results.json) retains every app attempt and the earlier
ergonomics-only and original dogfood runs. No credentials, raw responses or model answers
are stored in that report. Nothing was pushed or posted.

**Six of six live applications passed: 39 request boundaries /
101.76 seconds.** All runs were sequential. Live chat uses four turns;
offline chat covers twelve turns. No failed app attempts were discarded.

| Application | Outcome | Request boundaries | Seconds |
| --- | --- | ---: | ---: |
| research_assistant | passed | 5 | 12.39 |
| code_reviewer | passed | 6 | 14.60 |
| approval_workflow | passed | 11 | 33.54 |
| long_chat | passed | 11 | 35.25 |
| parallel_pipeline | passed | 5 | 4.50 |
| provider_switch | passed | 1 | 1.48 |

The counts are observed model/summary request-boundary invocations, not billing records.
One attempt is allowed per boundary. The pipeline includes one expired-deadline probe
rejected before dispatch, so live model HTTP attempts are at most **38**. Credential refresh
is not a model call. Only Codex ran live; the other six provider-switch routes stayed mocked.

## Behaviors verified

- **research_assistant:** synthetic search/fetch tools ran; the staged agent's own typed
  validator accepted citations and its pilot recommendation under fail-closed validation.
  No separate writer is needed. Extension summaries are asserted offline; live extension
  count remains uninstrumented (`null`).
- **code_reviewer:** read both disposable files, submitted a structured empty-input finding,
  handled a 5 ms lint timeout and applied 1,200-character output caps.
- **approval_workflow:** saved a pending question, resumed YES in a new process, and forked
  an independent NO rejection. READY/HOLD outcomes and blocked dependents passed.
- **long_chat:** four `send()` turns, three observed compactions, fresh-process pending
  response resume and memory recall passed. Grouped context budgets and public counters
  replace direct history edits. Offline verification covers twelve turns.
- **parallel_pipeline:** blocking `run(parallel=True, concurrency=2)` preserved both fan-in
  inputs. Borrowed HTTPX client configuration survived clones and shutdown left it open.
  Expired-deadline and offline pre-dispatch SIGINT checks passed.
- **provider_switch:** Codex completed and resumed its saved post-model response without
  an extra request. Six other routes used provider-native mocked envelopes.

## Offline, quality and parity

- All six installed-wheel apps passed offline; the subprocess app suite plus cancellation
  reports **7 passed in 2.01 s**.
- Final source `pytest`: **1,386 passed, 2 opt-in skips**, no xfails or warnings.
- `python scripts/quality.py`: passed Ruff, compilation, Pyright (zero errors/warnings),
  the same full suite, coverage inventory/matrix, fresh wheel package data and isolated imports.
- **100% lines (9,772/9,772)**, **97.1816% branches (2,655/2,732)**,
  **99.3842% combined**. All **129 modules / 16 areas** pass every raised hardening floor.
  Global 99% line / 95% branch / 99% combined floors and every per-area floor are unchanged.
- All three original DOGFOOD reproductions are assertive passing tests.
- All 42 execution-parity traces and seven offline provider reasoning probes pass.
  Current detailed evidence is in [merge-validation.md](../../docs/v2/merge-validation.md).

## Codex matrix

The subsequent installed-wheel matrix passed **14 cases**, failed none, and skipped
`reasoning_tools` with `reasoning_not_exposed`. Its runtime fingerprint matches the app
wheel. Live reasoning replay is not established by that skip; seven provider routes are
covered offline. Each case has a 60-second limit.

## Reproduce

Use the [clean installed-wheel setup](README.md). The measured app environment was
`artifacts/installed-runtime`; invoke its Python for these commands:

```bash
export PATH="$PWD/.venv/bin:$PATH" VIRTUAL_ENV="$PWD/.venv"
WHEEL_PYTHON="$PWD/artifacts/installed-runtime/bin/python"
"$WHEEL_PYTHON" -m pytest -q src/IkaTest/test_dogfood_apps.py
"$WHEEL_PYTHON" -m examples.apps.research_assistant.main --live --use-local-login --model gpt-5.6-sol
"$WHEEL_PYTHON" -m examples.apps.code_reviewer.main --live --use-local-login --model gpt-5.6-sol
"$WHEEL_PYTHON" -m examples.apps.approval_workflow.main --live --use-local-login --model gpt-5.6-sol
"$WHEEL_PYTHON" -m examples.apps.long_chat.main --live --use-local-login --model gpt-5.6-sol --turns 4
"$WHEEL_PYTHON" -m examples.apps.parallel_pipeline.main --live --use-local-login --model gpt-5.6-sol
"$WHEEL_PYTHON" -m examples.apps.provider_switch.main --live --use-local-login --model gpt-5.6-sol
python scripts/live_matrix.py --provider codex --use-local-login --model gpt-5.6-sol
```

Use `--offline` for synthetic responses. Ordinary apps have 90-second operation scopes
and one request attempt; chat has 180-second sessions with 30-second turn scopes.
Approval/chat child processes have their own scopes; the app runner bounded each whole
live application to 240 seconds. `IKACORE_APP_METRICS=/absolute/path/file.jsonl` records
boolean request-start rows, including model, summary and compaction boundaries.

## Historical evidence and limits

The ergonomics-only run at `ed55da5` used 44 boundaries / 126.00 seconds. The original
dogfood run at `aca8c53` included workarounds and an initial failed chat attempt, totaling
74 boundaries. Both histories remain in the JSON. Model choices and call sequences vary;
these runs are not a performance benchmark.

This is one provider/model/account sample. See the [resolution table](../../docs/v2/dogfood-report.md#resolution--ergonomics-implementation),
[migration guide](../../docs/v2/README.md#breaking-changes-and-migration) and
[API reference](../../docs/API.md) for the final supported contracts.
