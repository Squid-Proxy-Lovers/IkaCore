# Contributing

## Contribution License

Unless a separate agreement states otherwise, contributions intentionally submitted for inclusion in IkaCore are licensed under the [Apache License, Version 2.0](LICENSE), as described in Section 5 of the license. Submit only contributions you have the right to license on these terms, and preserve any applicable third-party license and attribution notices.

## Local Setup

```bash
python -m venv .venv
export PATH="$PWD/.venv/bin:$PATH" VIRTUAL_ENV="$PWD/.venv"
python -m pip install -e ".[test,dev]"
```

Install the secret-scanning hook once per clone. CI runs the same pinned
[Gitleaks](https://github.com/gitleaks/gitleaks) scan on every pull request and push to `main`:

```bash
pip install pre-commit
pre-commit install
```

Keep credentials in environment variables; never commit `keys.cfg`, `.env` files,
`auth.json`, raw model transcripts, or snapshot databases.

## Test Commands

Keep that environment active for every command, including Ruff and Pyright, so they
resolve the same Python and HTTPX installation. Run the main suite:

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

### Compatibility and deadline checks

Run the full suite on Python 3.11, 3.12 and 3.13 with both HTTPX 0.27.0 and the
current supported HTTPX release. Keep each combination in its own ignored venv:

```bash
python3.11 -m venv artifacts/compat/py311-httpx027
artifacts/compat/py311-httpx027/bin/python -m pip install -e ".[test]" "httpx==0.27.0"
artifacts/compat/py311-httpx027/bin/python -m pytest -q
```

Repeat for each interpreter/version pair, invoking that venv's Python explicitly.
Run heavy checks sequentially. The real HTTP/HTTPS deadline and pool-ownership
regressions use guarded local sockets; repeat this group ten times for release checks:

```bash
pytest -q src/IkaTest/test_transport_deadline_accuracy.py \
  src/IkaTest/test_transport_loopback.py src/IkaTest/test_transport_pool_ownership.py
```

### Live provider validation

Normal CI is offline. To validate configured providers once, install build/test tools
and run the wheel-backed matrix:

```bash
python -m venv .venv
export PATH="$PWD/.venv/bin:$PATH" VIRTUAL_ENV="$PWD/.venv"
python -m pip install -e ".[test,dev]"
python scripts/live_matrix.py --all-configured
```

The helper builds the current source into a wheel, installs it with **runtime dependencies
only** in a temporary venv, runs the checks, and removes that venv. The underlying
`scripts/live_provider_check.py` rejects editable/source installations. It can also be
invoked with an existing runtime wheel installation. `live_codex_check.py` remains a
thin compatibility entry point.

Credentials come only from environment variables. Set a model for each configured
provider using the variable below, `--model` for one provider, or `--models` with a JSON
provider-to-model mapping. Models are explicit to avoid relying on changing provider
defaults. A missing credential prints `not configured` and records all cases as skipped;
a credential with no model prints `model required` and makes the command fail.

| Provider ID | Credential variable | Model variable | Runtime route |
| --- | --- | --- | --- |
| `codex` | `CODEX_BEARER` | `IKACORE_LIVE_CODEX_MODEL` | Codex Responses |
| `openai-responses` | `OPENAI_API_KEY` | `IKACORE_LIVE_OPENAI_RESPONSES_MODEL` | OpenAI `/v1/responses` |
| `openai-chat` | `OPENAI_API_KEY` | `IKACORE_LIVE_OPENAI_CHAT_MODEL` | OpenAI `/v1/chat/completions` |
| `anthropic` | `ANTHROPIC_API_KEY` | `IKACORE_LIVE_ANTHROPIC_MODEL` | Anthropic `/v1/messages` |
| `deepseek` | `DEEPSEEK_API_KEY` | `IKACORE_LIVE_DEEPSEEK_MODEL` | DeepSeek Chat |
| `gemini` | `GEMINI_API_KEY` | `IKACORE_LIVE_GEMINI_MODEL` | Gemini `generateContent` |
| `openrouter` | `OPENROUTER_API_KEY` | `IKACORE_LIVE_OPENROUTER_MODEL` | OpenRouter Chat |
| `zai` | `ZAI_API_KEY` | `IKACORE_LIVE_ZAI_MODEL` | OpenAI adapter at `https://api.z.ai/api/coding/paas/v4/chat/completions`, GLM thinking enabled |

For example, use an existing local ChatGPT login explicitly for Codex only:

```bash
PATH="$PWD/.venv/bin:$PATH" VIRTUAL_ENV="$PWD/.venv" python scripts/live_matrix.py \
  --provider codex --use-local-login --model gpt-5.6-sol
```

`--use-local-login` invokes the existing Codex credential helper and may refresh stored
credentials; it is never implicit, including with `--all-configured`. It uses `CODEX_HOME`
when configured. See the [authentication documentation](https://learn.chatgpt.com/docs/auth).
Never put credential values in CLI arguments, model mappings, files, or workflow inputs.

Select providers with repeated `--provider`, or helper option `--providers codex,zai`.
`--case CASE` is repeatable. Every selected provider gets the following matrix:

| Cases | Providers / capability |
| --- | --- |
| `sync_text`, `async_text` | All; exact small answer + positive token accounting |
| `sync_tools`, `async_tools` | All; synthetic tool execution and exact tool-result replay |
| `simple_agent`, `staged_agent` | All; termination and two-stage snapshot evidence |
| `fresh_process_resume` | All; replay a saved post-model turn in a fresh process with network blocked and usage unchanged |
| `async_cancellation` | All; cancel after real response headers, verify response closure and caller-client ownership |
| `reasoning_tools` | Z.AI GLM, DeepSeek `reasoner`, OpenAI reasoning models, Codex; require parsed tool-turn reasoning to survive into the next payload |
| `streaming_sse` | Codex; require the live SSE collector to parse a completed output |
| `usage_accounting` | All; positive provider-reported usage |
| `deadline` | All; zero request-control budget must fail before network access |
| `transient_retry` | All; synthetic HTTP 503 then 200 via MockTransport, no live retry injection |
| `compaction` | All; controlled small context budget, real summarization, post-compaction snapshot, coherent agent completion |
| `workflow_hitl` | All; two dependent nodes, ask_user pause, resume_input, random value propagation through a named synthetic context field |

Unsupported cases are explicit skips, not compatibility claims. The current runtime has
no SSE path outside Codex and no Anthropic extended-thinking configuration or signed
thinking-block replay. Gemini/OpenRouter reasoning is not exercised by this harness.
If a reasoning model exposes no tool-turn reasoning, `reasoning_tools` records
`reasoning_not_exposed`; preservation remains unverified. Use a reasoning model for
that case (for example, DeepSeek's reasoner rather than chat model).

Each case runs in a separate process with a **60-second wall-clock deadline** and
`request_controls(timeout=60, max_retries=1)`. `--case-timeout 1..120` changes the limit.
There is one live attempt per request, tiny prompts, at most three simple-agent steps/tool
rounds, two steps per stage, and no step/stage extensions. The retry case alone permits two **synthetic**
attempts. Generation is capped at 512 output tokens where supported; summarization
uses the runtime's 2000-token cap. Codex accepts neither cap, so this is not a strict
spend/token bound. Failed text connectivity checks skip remaining cases for that provider.

Reports at `artifacts/live/<provider>.json` contain a schema version, model/endpoint,
package version, git revision, installed-runtime SHA-256, requested cases, sanitized
outcomes, allowlisted failure checkpoint IDs, elapsed times, and numeric usage. `usage_complete: false` marks cases whose
full billing usage is unavailable (especially cancellation and failures). Compaction
usage includes its summary request. Child stdout/stderr and exception messages are
discarded; request dumps, GLM reasoning logs and TLS key logs are disabled. Synthetic
snapshots live only in temporary directories. Compaction retention and workflow dataflow
use random fixture values that are absent from the final answering instruction. No credentials or raw model output belong
in reports. Generate the Markdown summary independently with:

```bash
python scripts/live_summary.py --directory artifacts/live
```

The helper creates `summary.md` automatically. Exit status is nonzero for failed cases
or missing model configuration; an entirely unconfigured run succeeds with explicit
skips. Inspect skips and requested/completed cases before accepting a run as validation.
The optional pytest smoke entry point uses the same installed-wheel harness:
`IKACORE_LIVE_PROVIDER_TESTS=1 pytest src/IkaTest/test_live_provider_smoke.py`.

The manual **Manual live providers** workflow (`.github/workflows/live-providers.yml`)
uses `workflow_dispatch` only, with `providers` (comma-separated IDs or `all-configured`)
and `models` (JSON mapping) inputs. Configure the credential variables above as repository
**secrets**, and model variables as repository **vars**, or supply models in the dispatch
input. It has read-only repository permissions, pinned actions, a bounded job timeout,
and uploads only sanitized reports. It never runs on PRs or a schedule.

The latest installed-wheel matrix recorded **14 passed, 0 failed, 1 skipped** with
`gpt-5.6-sol` and local login on **2026-10-18 UTC**, runtime/test revision `179b52e`.
`reasoning_tools` skipped because the service exposed no tool-turn reasoning. Each
case was bounded to 60 seconds. See [the merged validation report](docs/v2/merged-validation.md)
for results and revision labels; the earlier `21184cc` harness run used the pre-refactor API.

Offline repro `python scripts/live_replay_repro.py` shows that Codex and OpenAI
Responses parse synthetic tool-turn reasoning but omit it from the next request, and
Anthropic neither enables extended thinking nor preserves its signed thinking blocks.
Z.AI is a positive replay control. These are existing runtime gaps, left unchanged by
the live harness work; a live reasoning skip does not validate those paths.

The historical **Codex 8/8** result on 2026-10-17 used `gpt-5.6-sol`, local Codex login,
and the installed 2.0.0 wheel at runtime revision `1f2bab0`; `dc9efbd` retains that
validated runtime baseline. An earlier 8/8 run at `183ce24` predates review fixes.
These results establish that model/endpoint/account combination only. Other providers
remain unverified until their keys and models are configured and their matrix passes.

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


The installed-wheel provider benchmark measures all seven registered routes, with 10,
100 and 1,000 messages, and times payload construction separately from JSON encoding:

```bash
/tmp/ika-runtime-bench/bin/python scripts/provider_benchmark.py \
  --output artifacts/benchmarks/providers.json
```

Use `--baseline /path/to/previous-providers.json` for before/after comparison. It rejects
changed payload checksums before comparing timings. The test suite also checks all 21
fixture checksums against the pre-optimization runtime. These are synthetic transcript
checks, supplemented by existing tool, reasoning, history, image and retry regressions;
they do not certify live provider compatibility. The offline suite blocks socket
connections unless `IKACORE_LIVE_PROVIDER_TESTS=1` is explicitly set.
