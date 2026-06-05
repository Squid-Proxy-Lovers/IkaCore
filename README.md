# IkaCore

IkaCore is a Python agent framework for building staged agents, subagent tool delegation, and multi-agent workflows across multiple LLM providers.

The current runtime supports:

- `IkaBaseAgent` for single-agent execution
- `IkaStage` for bounded staged execution
- `IkaWorkflow` for multi-agent graphs
- OpenAI, Anthropic, DeepSeek, Gemini, OpenRouter, and Codex backends
- checkpoint/resume and interruptible HITL flows
- optional short-term and long-term memory helpers

## Install

```bash
pip install -e .
```

For development:

```bash
pip install -e .[test]
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

## Main Concepts

- `IkaBaseAgent`: provider config, prompt, tools, memory, checkpoints, optional stages
- `IkaTools`: Python callables exposed to the model as structured tool calls
- `IkaStage`: phase-specific prompt, tool set, and optional model overrides
- `IkaWorkflow`: graph runner with `next` and tool-only `child` edges

## Current Semantics

- OpenAI uses the Responses API by default
- Set `use_responses_api=False` to use Chat Completions
- Stage execution preserves prior transcript context
- `child` workflow edges wire subagents into a stage and do not auto-run as standalone nodes
- HITL uses interrupt/resume rather than blocking terminal input

## Documentation

- [Usage Guide](docs/USAGE.md)
- [Architecture Notes](docs/indepth.md)
- [Memory Notes](src/IkaMem/docs/README.md)

Codex support is documented in the [Provider Selection](docs/USAGE.md#provider-selection) section. IkaCore treats Codex auth as opt-in: pass a bearer token as `api_key`, or call `IkaModel.codex.codex_auth.get_bearer()` to read and refresh the Codex CLI token from `~/.codex/auth.json`.

## Examples

- `examples/example_fileagent.py`
- `examples/example_stages.py`
- `examples/example_hitl.py`
- `examples/example_workflow.py`

## Development Notes

This repo has active cleanup work. The docs now reflect the current runtime behavior, but a few internal hygiene issues still remain, especially legacy `sys.path` mutation inside some modules and examples.
