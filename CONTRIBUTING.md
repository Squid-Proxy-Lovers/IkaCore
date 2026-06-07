# Contributing

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

- keep `python scripts/quality.py` passing
- keep packaging metadata and docs accurate
- add regression tests for behavior changes before modifying provider logic
- keep production functions and classes below the structural size gate enforced in `src/IkaTest/test_quality_gates.py`
