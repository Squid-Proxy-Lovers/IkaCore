# Codex application usage results

Date: **2026-10-18T06:21:34.181236+00:00**. Model: **gpt-5.6-sol**. All runs passed
`--live --use-local-login --model gpt-5.6-sol`; local login access was explicitly opted in.
Runtime: a fresh, non-editable IkaCore 2.0.0 wheel from implementation revision
`ed55da5928451a26e600d59b3266fef33bc92d17` on `v2-ergonomics` (base `49dbeb3`),
Python 3.13.15 / HTTPX 0.28.1. Imports were verified under this worktree's
`.venv/lib/python3.13/site-packages`, without editable imports or PYTHONPATH.

Wheel SHA-256: `7d946996cbe3fbfebef67609860c5ca5713e644f37b3d39dc3330aef79ee451e`.
Detailed sanitized evidence is in [live-results.json](live-results.json).
No credentials, prompts, raw responses or model answer text are retained there.

**Six of six applications passed, using 44 request-boundary invocations.**
Runs were sequential. The live chat uses four turns to keep calls modest; the installed-wheel
offline suite still exercises twelve turns. No failed live attempts were discarded.

| Application | Outcome | Request-boundary invocations | Wall time (s) |
| --- | --- | ---: | ---: |
| research_assistant | passed | 5 | 17.14 |
| code_reviewer | passed | 6 | 13.85 |
| approval_workflow | passed | 11 | 30.65 |
| long_chat | passed | 16 | 55.76 |
| parallel_pipeline | passed | 5 | 6.87 |
| provider_switch | passed | 1 | 1.73 |

These are observed model/summary request-boundary counts, not provider billing records.
One request attempt was allowed per boundary. The pipeline includes one expired-deadline
probe rejected before dispatch, so actual model HTTP attempts are at most **43**.
Credential refresh is not a model call. Only Codex ran live; all other provider routes
remained mocked.

## What passed with the new APIs

- **research_assistant:** actual synthetic search/fetch tools ran; the staged agent's own
  typed final check accepted both citations and its pilot recommendation under
  `fail_closed=True`. The separate writer workaround is gone. A stage-extension summary
  is asserted offline; the live extension count remains uninstrumented (`null`).
- **code_reviewer:** read both disposable files, submitted a structured empty-input finding,
  survived a 5 ms lint timeout and used 1,200-character output caps. Library diagnostics
  stay quiet by default; offline checks assert timeout feedback and capped payloads.
- **approval_workflow:** a process saved the pending question, another answered YES, and
  `workflow.fork_from_snapshot(..., resume_input="NO")` answered an independent rejection.
  READY/HOLD outcomes and blocked dependents passed; nothing was published.
- **long_chat:** four `send()` turns, **three observed compactions**, fresh-process pending
  response resume and memory recall all passed. The token budget is grouped under
  `ExecutionLimits(context_budget=700)` and the app reads `compaction_count`; it no longer
  edits `first_input` or uses stage history entries to simulate user turns. Offline testing
  exercises twelve turns and the public adapter.
- **parallel_pipeline:** blocking `run(parallel=True, concurrency=2)` retained both cost
  and risk at fan-in. `ProviderConfig(client=client)` preserved a caller-owned synchronous
  HTTPX client across clones and worker shutdown; no subclass/detach workaround remains.
  The expired deadline was rejected. Cooperative pre-dispatch SIGINT passed offline;
  arbitrary in-flight live cancellation is not claimed.
- **provider_switch:** Codex completed and resumed its saved post-model response. Six other
  routes passed provider-native mocked envelopes with the same public agent contract.

## Offline and quality evidence

Final plain `pytest -q`: **1,075 passed / 1 skipped**, no xfails, **19.37 seconds**.

- Installed-wheel app tests: **7 passed in 1.64 s** (all six apps plus SIGINT pre-dispatch).
- All three DOGFOOD regression reproductions pass with xfail markers removed and original
  assertions intact. HTTPX fixture request metadata was corrected so assertions execute.
- `python scripts/quality.py`: **passed**, including Ruff, Pyright (zero errors), compilation,
  **1,075 passed / 1 skipped** with coverage, all **127 modules / 16 areas** with zero gate
  failures, fresh-wheel package data/module inventory and isolated imports.
- Combined coverage: **94.80%** (line **96.46%**, branch **88.82%**), with all floors unchanged.
  Workflow branch coverage is **93.75%**, above its unchanged 90% floor.

## Reproduce

Follow [clean-wheel setup](README.md), then run each module from the repository root:

```bash
python -m examples.apps.research_assistant.main --live --use-local-login --model gpt-5.6-sol
python -m examples.apps.code_reviewer.main --live --use-local-login --model gpt-5.6-sol
python -m examples.apps.approval_workflow.main --live --use-local-login --model gpt-5.6-sol
python -m examples.apps.long_chat.main --live --use-local-login --model gpt-5.6-sol --turns 4
python -m examples.apps.parallel_pipeline.main --live --use-local-login --model gpt-5.6-sol
python -m examples.apps.provider_switch.main --live --use-local-login --model gpt-5.6-sol
```

Omit `--live` to run offline. Chat defaults to twelve turns. The shared session bounds
ordinary apps to 90 seconds and one request attempt. Chat sessions use 180 seconds with
30-second turn scopes; approval/chat child processes have their own scopes. Tool/model
limits also apply. `IKACORE_APP_METRICS=/absolute/path/to/file.jsonl` records boolean
request-start rows, including summary/compaction requests, without request bodies.

## Historical comparison

The original dogfood run on `aca8c53` needed app workarounds, a separate validated writer
and an initial failed chat attempt; its seven attempts totaled 74 boundaries. That full
record is preserved under `previous_dogfood_run` in the JSON. It is separate from this
44-boundary rerun. The reduction also reflects four live chat turns versus twelve earlier;
it is not a runtime performance benchmark.

This is one provider/model/account sample. See every item's
[resolution](../../docs/v2/dogfood-report.md#resolution--ergonomics-implementation), the
[migration table](../../docs/v2/README.md#breaking-changes-and-migration), and the
[tested API reference](../../docs/API.md) for the current contracts.
