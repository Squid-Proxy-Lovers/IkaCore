> Historical development evidence. Counts and API examples apply only to the recorded revision.

# Merged v2 implementation validation

The final runtime and tests are at `179b52e4076815dc74033c6622116f2d993cb9ce`,
following merge `6ef53d4539cfe56b0e2d840cfebbdba40b13f6ff`. This report and the unified
documentation are a documentation-only follow-up. Version 2.0.0 remains unreleased
and intentionally breaking against v1. Source changes and retained validation artifacts
are in `/private/tmp/ikacore-dedupe`; nothing was pushed or posted on GitHub.

## Integration changes

- Ported `Probe.agent()` in `scripts/live_provider_check.py` to `ProviderConfig`,
  `ExecutionLimits`, `ExecutionPolicy` and silent `OutputConfig`. Provider routing,
  token caps, three-step/tool budgets and disabled extensions retain their intended
  values. `live_codex_check.py` remains the thin wrapper.
- Expanded offline probes to check grouped construction for all eight routes and
  exercise simple/staged completion and fresh-process post-model resume, alongside
  the existing tools, compaction and workflow HITL probes. The historical behavioral
  fixture adapter stays test-only; no flat constructor compatibility was added.
- Added controlled JSON and Codex SSE integration tests for arbitrary context variables,
  request controls, runtime options, scoped output, workflow instance identity, snapshot
  hooks and `RuntimeControl`. Both worker-raised transport pauses and normal post-model
  pauses reach the snapshot dispatcher; saved responses resume without another request
  or double-counted usage. Caller clients remain open and parent scopes restore.
- Found and fixed a semantic integration bug: `_SyncRequestAbort.abort()` started its
  custom response-close thread without copying context. A delayed close observed
  `('outside', 9, None)` instead of the request's `('inside', 1, 0.7)` and lost its output
  subscriber. `copy_context().run` now preserves the scope even after the caller returns.
  The new regression fails on `5b85054` and passes with `179b52e`.
- Reviewed the merged request-control setters and worker helpers, and extended required
  coverage scenarios without changing any floors or weakening existing assertions.
- Unified the unreleased changelog, grouped API examples, request/snapshot control
  explanations, live-testing instructions, architecture map and revision-labeled
  measurements. Fixed the migration table's misplaced separator and stale live links.

## Offline results

Measured on macOS arm64, CPython 3.13.15, HTTPX 0.28.1 and httpcore 1.0.9:

| Check | Result |
| --- | --- |
| `pytest -q` | 973 passed, 1 skipped |
| `python scripts/quality.py` | Passed: Ruff, compilation, Pyright, coverage, wheel inventory/imports |
| Pyright | 0 errors, 0 warnings |
| Line coverage | 96.3850% |
| Branch coverage | 88.1955% |
| Combined coverage | 94.6239% |
| Coverage inventory | 122 modules, 16 areas, 119 required area/scenario checks; 117 unique selectors |
| Coverage gate failures | 0; global and per-area floors unchanged |
| Excluded lines | 194 |
| Fresh execution traces against `dc9efbd` | 42/42 identical |
| HTTP/HTTPS deadline/ownership suites | 60 tests per round, 10 consecutive rounds, 600 passes |

The single main-suite skip is the opt-in installed-wheel live smoke entry point.
Parity uses the unmodified `dc9efbd92149e98bed947318c50e90fd77e73f3d` source exported
inside ignored artifacts and the same trace script for both versions. Results,
provider call sequences, usage, budgets, extensions, transitions, HITL and snapshot
boundaries all match. Random snapshot identities/timestamps are normalized.

Repeated transport checks cover HTTP/HTTPS trickled headers/body, accept/TLS stalls,
Codex SSE, explicit and check-only cancellation, worker cleanup and caller-pool races,
as well as the uncontrolled-path regression. The final runtime was tested again after
the cleanup-context fix; earlier pre-fix repetitions are separate artifacts.

## Python / HTTPX matrix

Every cell ran the complete pytest command sequentially in a separate venv under
`artifacts/merge-validation/matrix/`, with `.[test]` and the selected HTTPX version.
All cells used httpcore 1.0.9.

| Python | HTTPX | Passed | Skipped | Failures |
| --- | --- | ---: | ---: | ---: |
| 3.11.16 | 0.27.0 | 967 | 7 | 0 |
| 3.11.16 | 0.28.1 | 967 | 7 | 0 |
| 3.12.14 | 0.27.0 | 967 | 7 | 0 |
| 3.12.14 | 0.28.1 | 967 | 7 | 0 |
| 3.13.15 | 0.27.0 | 967 | 7 | 0 |
| 3.13.15 | 0.28.1 | 967 | 7 | 0 |

The seven skips are six dev-only lint-policy fixtures (Ruff is absent in test-only
venvs) plus the opt-in live smoke test. No runtime regression is skipped. The main
`.[test,dev]` quality run above exercises all six lint fixtures. Python 3.13.15 came
from `~/.pyenv/versions`; 3.11.16 and 3.12.14 used the available standalone interpreters
under `/private/tmp/ikacore-r3-python/`, creating only new venvs in this worktree.
HTTPX 0.28.1 was the current supported release resolved during installation.

## Wheel and live validation

The fresh wheel built after the compatibility matrix passes inventory and isolated
imports: **217,960 bytes / 212.85 KiB** compressed, **706,313 bytes / 689.76 KiB**
unpacked. It ships one direct dependency (`httpx`), 122 runtime Python modules and
17,291 physical Python lines. No runtime dependency was added.

On **2026-10-18 UTC**, the installed 2.0.0 wheel at runtime revision `179b52e`
passed the Codex-only `gpt-5.6-sol` matrix: **14 passed, 0 failed, 1 skipped**.
Each case had a 60-second subprocess/request budget and one live attempt per request;
the retry case used only synthetic responses. The installed runtime SHA-256 was
`050a6481a34d635065afe2babd9450f9cc1edc276aa5cb63c8f6521ee0f2ac8a`.

Passed: sync/async text, sync/async tools, simple agent, staged agent, fresh-process
resume, async cancellation, SSE parsing, usage accounting, expired deadline rejection,
synthetic transient retry, compaction and workflow HITL. Resume made no network call
and preserved usage; cancellation closed the response and kept the caller client open.
Compaction produced one saved boundary and retained the fixture; workflow resume
preserved the dependent node's context and answer.

`reasoning_tools` skipped with `reasoning_not_exposed`: no tool-turn reasoning was
exposed by the live service, so reasoning replay remains unverified. This skip does
not establish support for the existing reasoning-replay gaps documented in CONTRIBUTING.
Cancellation and skipped reasoning cases mark their usage incomplete. Reports contain
only sanitized outcomes and numeric usage, with no credentials or raw model output.

The existing limits remain: OS/Python scheduling applies; blocking DNS, pre-connect
system calls or uncooperative custom code may retain daemon workers until they return.
The caller wait remains bounded. Real proxy handshakes, real HTTP/2, hosted memory
backends, other live providers/models and long-duration application behavior were not
qualified by this validation.

## Reproduction and evidence

```bash
python -m venv .venv
export PATH="$PWD/.venv/bin:$PATH" VIRTUAL_ENV="$PWD/.venv"
python -m pip install -e ".[test,dev]"
pytest -q
python scripts/quality.py
python scripts/execution_parity.py --output artifacts/merge-validation/parity/final.json
# Repeat this transport group ten times, sequentially:
pytest -q src/IkaTest/test_transport_deadline_accuracy.py \
  src/IkaTest/test_transport_loopback.py src/IkaTest/test_transport_pool_ownership.py
python scripts/live_matrix.py --provider codex --use-local-login --model gpt-5.6-sol \
  --case-timeout 60 --output-dir artifacts/merge-validation/live
```

Use the separate-venv instructions in [CONTRIBUTING](../../../../CONTRIBUTING.md#compatibility-and-deadline-checks)
for each compatibility cell. All heavy checks in this report ran sequentially.
The live run used a private temporary copy of the existing login so refreshes could
stay inside ignored artifacts; that copy is removed after the command.

Local evidence (gitignored):

- `artifacts/pytest-final.log`, `artifacts/quality-merged-final.log`
- `artifacts/quality/{coverage.json,coverage-matrix.json,tests.xml,ikacore-2.0.0-py3-none-any.whl}`
- `artifacts/merge-validation/parity/{dc9efbd.json,final.json,comparison.json}`
- `artifacts/merge-validation/cleanup-context-{before,after}.log`
- `artifacts/merge-validation/deadline-final/{results.json,round-*.log}`
- `artifacts/merge-validation/matrix/{results.json,*.log,*.xml}`
- `artifacts/merge-validation/{wheel.log,live.log,live/codex.json,live/summary.md,report.json}`

## Local commits

- `5b850540b85063b176c419840088ef7f402c5979`: grouped live probes and controlled-runtime integration coverage.
- `179b52e4076815dc74033c6622116f2d993cb9ce`: context-preserving abort cleanup and reproduced regression.
- The documentation-only commit containing this report unifies the guides and records
  the final measurements. Every commit ends with the requested Claude Opus 5.5 co-author trailer.
