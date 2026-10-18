# IkaCore

IkaCore is a Python agent framework for building staged agents, subagent tool delegation, and multi-agent workflows across multiple LLM providers.

The current runtime supports:

- `IkaBaseAgent` for single-agent execution
- `IkaStage` for bounded staged execution
- `IkaWorkflow` for multi-agent graphs
- OpenAI, Anthropic, DeepSeek, Gemini, OpenRouter, and Codex backends
- snapshot/resume and interruptible HITL flows
- optional memory integration through ordinary tools

Version 2.0.0 is unreleased and intentionally breaks compatibility with v1. See the [migration guide](docs/v2/README.md) before adopting it.

## Install

For application evaluation from this unreleased checkout, install a built wheel:

```bash
python3 -m venv .venv
export PATH="$PWD/.venv/bin:$PATH" VIRTUAL_ENV="$PWD/.venv"
python -m pip install build
python -m build --wheel
python -m pip install dist/ikacore-2.0.0-py3-none-any.whl
python -m examples.apps.research_assistant.main --offline
```

See [six runnable applications](examples/apps/README.md), their [live results](examples/apps/LIVE_RESULTS.md),
and the [new-user findings](docs/v2/dogfood-report.md).

For development:

```bash
python -m venv .venv
export PATH="$PWD/.venv/bin:$PATH" VIRTUAL_ENV="$PWD/.venv"
python -m pip install -e ".[test,dev]"
pytest
```

## Quick Start

```python
from IkaCore import IkaBaseAgent

agent = IkaBaseAgent(
    name="assistant",
    description="Simple example agent",
    prompt="Answer the user request in a short paragraph.",
    model_id="gpt-4o-mini",
    api_key="...",
)

result = agent.execution()
print(result["final_message"])
```

Common calls remain short: `IkaBaseAgent("helper", prompt="Summarize the task", model_id="gpt-4o-mini", api_key=key)`.
Advanced settings use frozen `ProviderConfig`, `PromptConfig`, `ExecutionLimits`,
`ExecutionPolicy`, `AgentTopology`, and `OutputConfig` values. For example:

```python
from IkaCore import ExecutionLimits, ExecutionPolicy, OutputConfig

agent = IkaBaseAgent("helper", prompt="Summarize the task", model_id="gpt-4o-mini", api_key="...",
                     limits=ExecutionLimits(maxsteps=10),
                     policy=ExecutionPolicy(max_step_extensions=0), output=OutputConfig(cli=False))
```

`agent.config` holds configured defaults; `agent.state` holds run budgets, history and
usage. `OutputConfig(subscribers=(callback,))` receives structured `RuntimeEvent`
values. Memory backends are explicitly adapted with `IkaMem.agent_tools.MemoryTools`.
See the [usage guide](docs/USAGE.md) for executable setup examples.

Bound provider requests across retries with a scoped deadline:

```python
from IkaModel import request_controls

with request_controls(timeout=60, max_retries=1):
    result = agent.execution()
agent.shutdown()
```

Controlled synchronous JSON and Codex SSE requests bound the caller's wait using
an absolute deadline. Their workers preserve output subscribers and snapshot controls;
caller-owned HTTPX clients remain usable. See the [request control limits](docs/v2/README.md#retained-controls).

## Main Concepts

- `IkaBaseAgent`: frozen configuration, prompt, tools, snapshots, optional stages
- `IkaTools`: Python callables exposed to the model as structured tool calls
- `IkaStage`: phase-specific prompt, tool set, and optional model overrides
- `IkaWorkflow`: graph runner with `next` and tool-only `child` edges

## Current Semantics

- OpenAI uses the Responses API by default
- Set `provider=ProviderConfig(use_responses_api=False)` to use Chat Completions
- Stage execution preserves prior transcript context
- `child` workflow edges wire subagents into a stage and do not auto-run as standalone nodes
- HITL uses interrupt/resume rather than blocking terminal input

## Documentation

- [Changelog](CHANGELOG.md)
- [Usage Guide](docs/USAGE.md)
- [Architecture Notes](docs/indepth.md)
- [IkaCore v2 draft: simplification and migration](docs/v2/README.md)
- [Memory Notes](src/IkaMem/docs/README.md)

Codex support is documented in the [Provider Selection](docs/USAGE.md#provider-selection) section. IkaCore treats Codex auth as opt-in: pass a bearer token as `api_key`, or explicitly import `get_bearer` with `from IkaModel.codex.auth import get_bearer` and call `get_bearer()` to read and refresh the Codex CLI token from `~/.codex/auth.json`.

## Examples

- `examples/example_fileagent.py`
- `examples/example_stages.py`
- `examples/example_hitl.py`
- `examples/example_workflow.py`

## Development Notes

This repo is maintained with `pytest` and `ruff`. Runtime modules and examples should import through the installed package rather than mutating `sys.path`; keep docs and examples aligned with public API changes.

## License

Copyright 2025-2026 IkaCore contributors.

IkaCore is licensed under the [Apache License, Version 2.0](LICENSE). It permits commercial use, modification, and redistribution, including use in proprietary applications, and includes an explicit contributor patent grant. Redistribution must comply with the license's notice and attribution requirements. See [NOTICE](NOTICE) for project attribution.
