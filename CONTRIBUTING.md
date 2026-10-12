# Contributing

## Contribution License

Unless a separate agreement states otherwise, contributions intentionally submitted for inclusion in IkaCore are licensed under the [Apache License, Version 2.0](LICENSE), as described in Section 5 of the license. Submit only contributions you have the right to license on these terms, and preserve any applicable third-party license and attribution notices.

## Local Setup

```bash
pip install -e ".[test,dev]"
```

## Test Commands

Run the main suite:

```bash
pytest
```

Run the full local quality gate:

```bash
python scripts/quality.py
```

To see the exact commands without running them:

```bash
python scripts/quality.py --list
```

Run the focused provider and workflow regressions:

```bash
pytest src/IkaTest/test_ika_base_agent.py
pytest src/IkaTest/test_workflow_semantics.py
pytest src/IkaTest/test_hitl_interrupt_resume.py
pytest src/IkaTest/test_provider_payload_builders.py
pytest src/IkaTest/test_provider_edge_helpers.py
```

Optional live provider smoke tests are skipped by default and require credentials. When `IKACORE_LIVE_PROVIDER_TESTS=1` is set, at least one provider must be fully configured or the smoke suite fails. OpenAI and DeepSeek have conservative default model ids; Anthropic, Gemini, and OpenRouter require explicit model env vars so the smoke command does not depend on stale defaults.

```bash
IKACORE_LIVE_PROVIDER_TESTS=1 OPENAI_API_KEY=... pytest src/IkaTest/test_live_provider_smoke.py
```

The same smoke suite is available in GitHub Actions through the `Live Provider Smoke` workflow. Configure provider API keys as repository secrets and provider model ids as repository variables before relying on the scheduled run.

Supported live smoke variables:

- `OPENAI_API_KEY`, optional `IKACORE_LIVE_OPENAI_MODEL`
- `ANTHROPIC_API_KEY`, required `IKACORE_LIVE_ANTHROPIC_MODEL`
- `DEEPSEEK_API_KEY`, optional `IKACORE_LIVE_DEEPSEEK_MODEL`
- `GEMINI_API_KEY`, required `IKACORE_LIVE_GEMINI_MODEL`
- `OPENROUTER_API_KEY`, required `IKACORE_LIVE_OPENROUTER_MODEL`

## Project Conventions

- keep runtime behavior aligned with tests
- prefer narrowly scoped changes over broad refactors
- update `docs/USAGE.md` when the public API changes
- preserve current workflow semantics:
  - `next` edges propagate context
  - `child` edges are stage wiring only
- treat subagent delegated input as untrusted content

## Current Cleanup Priorities

The [v2 simplification and migration guide](docs/v2/README.md) records the intended
runtime behavior, removed compatibility paths, and remaining cleanup candidates.
V1 compatibility is not a requirement. Document breaking changes and retain tests
that verify behavior and invariants.

- keep `python scripts/quality.py` passing
- keep packaging metadata and docs accurate
- cover changed provider/runtime semantics with meaningful regression tests
- keep production functions within the 70-line limit and modules within the 600-line limit
- group related behavior in cohesive classes; avoid mixin chains created solely to satisfy class-size limits

## Quality evidence

`python scripts/quality.py` runs Ruff correctness/async rules, a per-function cyclomatic
complexity limit of 15, Python compilation, Pyright, the behavioral suite, the coverage
matrix, and a fresh wheel build/import check. The dev extra includes the build backend
and wheel tooling, so the no-isolation packaging check also works in fresh Python 3.12+ environments. Representative invalid snippets are tested
against the actual Ruff configuration. Keep functions within 70 lines and modules within
600; large cohesive classes are allowed.

Coverage requires at least 91% combined coverage, **94% executable-line coverage**, and
**85% branch coverage**, plus independent floors for each of the 16 subsystems in
`scripts/coverage_manifest.json`. Every runtime module must belong to exactly one area.
Required scenarios must appear and pass in JUnit; skipped/missing scenarios fail the gate.
Generated reports include observed test contexts, exact missing line/branch locations,
excluded-line counts, and known gaps. Line/branch coverage does not measure model quality.

Reports are written to `artifacts/quality/coverage-matrix.md`, `coverage-matrix.json`,
`coverage.json`, and `tests.xml`. They are uploaded from CI on Python 3.11, 3.12 and 3.13
on Ubuntu, plus Python 3.11 with the minimum HTTPX 0.27.0 dependency. Live-provider tests remain opt-in; the offline CI records their skips.

The policy deliberately exempts constant `getattr`/`setattr` adapters, public async
`timeout` arguments, and typed dataclass-factory lambdas from style-oriented lint rules.
Mutable defaults, lost tasks, missing exception causes, blocking async calls, blind
exception assertions, and overly complex functions remain checked.

## Reproducing the offline benchmarks

Build/validate the wheel with the quality command first. Use a disposable runtime-only
environment so development tools do not inflate the dependency footprint:

```bash
python -m venv --without-pip /tmp/ika-runtime-bench
python -m pip --python /tmp/ika-runtime-bench/bin/python install artifacts/quality/*.whl
python scripts/benchmark.py --python /tmp/ika-runtime-bench/bin/python \
  --wheel artifacts/quality/*.whl \
  --samples 9 --iterations 100 --soak-iterations 1000
```

Requires Linux or macOS for the RSS measurement. The worker rejects real socket
connections and uses fixed data, HTTPX MockTransport, pure workflow fixtures, and local
SQLite. Model/network latency, optional memory backends and real downstream applications
are outside these measurements. Each sample uses a fresh interpreter; warm cases perform
three untimed warmups. The filesystem cache is not flushed. P95 is across subprocess batch
averages, not individual request latency. RSS includes the interpreter/measurement worker.

`artifacts/benchmarks/benchmark.json` stores raw samples, versions, installed distribution
sizes, RSS, wall times, and two phases of repeated-run allocation measurements. The report
records revision/dirty state and wheel/harness hashes, and verifies that measured runtime files match the wheel. `benchmark.md` is the readable table.
CI runs a smaller benchmark sample and uploads the evidence. Portable gates limit the
wheel to 300 KiB, unpacked wheel content to 1 MiB, and direct runtime requirements to one.
Timing results are reported rather than enforcing noisy cross-machine latency thresholds.
