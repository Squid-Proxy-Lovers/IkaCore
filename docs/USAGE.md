## IkaCore Usage Guide

For this v2 draft, review [breaking changes and migration](v2/README.md#breaking-changes-and-migration).
Provider replay, retry behavior, and workflow scheduling now use the v2 defaults.

IkaCore is a Python agent framework built around three primitives:

- `IkaBaseAgent`: an agent with provider config, tools, snapshots, and optional stages.
- `IkaStage`: a bounded execution phase with its own tool set and optional model overrides.
- `IkaWorkflow`: a graph runner for composing multiple agents.

This guide documents the current API in `src/`.

### Install

From the repo root:

```bash
pip install -e .
```

For test work:

```bash
pip install -e ".[test]"
pytest
```

### Minimal Agent

```python
from IkaCore import IkaBaseAgent

agent = IkaBaseAgent(
    name="assistant",
    description="Simple example agent",
    prompt="Answer the user request concisely.",
    model_id="gpt-4o-mini",
    api_key="...",
)

result = agent.execution()
print(result["final_message"])
```

Required constructor fields:

- `name`
- `description`
- `prompt`
- `model_id`
- `api_key`

### Tools

Tools are defined with `IkaTools`.

```python
from IkaCore import IkaBaseAgent, IkaTools
import os

list_files = IkaTools(
    name="list_files",
    description="List files in a directory",
    parameters={
        "directory_path": {
            "type": "string",
            "description": "Directory to inspect",
            "required": True,
        },
    },
    execute_function=lambda args: os.listdir(args["directory_path"]),
)

agent = IkaBaseAgent(
    name="file_agent",
    description="Reads local files",
    prompt="Inspect the requested directory and summarize what you find.",
    tools=[list_files],
    model_id="gpt-4o-mini",
    api_key="...",
)
```

Important `IkaTools` fields:

- `parameters`: JSON-schema-like argument metadata. If a property has `"required": True`, the generated tool schema marks it as required.
- `limit_calls`: `0` means unlimited.
- `required`: marks the tool as provider-required when the backend supports it. Use this sparingly.
- `parallel`: controls whether the tool can be grouped with parallel tool calls.

Every agent automatically gets an `agent_end` control tool. Staged agents also get stage-control tools when applicable.

### Stages

Stages let one agent execute in bounded phases with different tools and optional model overrides.

```python
import os
from pathlib import Path

from IkaCore import IkaBaseAgent, IkaStage, IkaTools

list_files = IkaTools(
    name="list_files",
    description="List files in a directory",
    parameters={"directory_path": {"type": "string", "required": True}},
    execute_function=lambda args: os.listdir(args["directory_path"]),
)
read_file = IkaTools(
    name="read_file",
    description="Read a UTF-8 text file",
    parameters={"file_path": {"type": "string", "required": True}},
    execute_function=lambda args: Path(args["file_path"]).read_text(encoding="utf-8"),
)

list_stage = IkaStage(
    name="discover",
    prompt="List the relevant files.",
    tools=[list_files],
    stage_max_step=5,
)

read_stage = IkaStage(
    name="read",
    prompt="Read the most relevant files and extract the important details.",
    tools=[read_file],
    stage_max_step=10,
)

agent = IkaBaseAgent(
    name="staged_agent",
    description="Performs a staged file analysis",
    prompt="Complete the staged workflow.",
    Stages=[list_stage, read_stage],
    model_id="gpt-4o-mini",
    api_key="...",
)
```

Current stage behavior:

- Stage context is preserved through the shared transcript in `message_history["messages"]`.
- The active stage prompt is refreshed per stage, so the model does not keep reusing stage 0's original user prompt.
- `stage_end` advances to the next stage.
- `change_stage` is optional and only available when `allowed_back_to` is defined.
- `ask_user` is available only on HITL stages and now interrupts execution instead of blocking on `input()`.

Useful `IkaStage` fields:

- `tools`
- `stage_max_step`
- `subagents`
- `allowed_back_to`
- `hitl`
- `memory_access`
- `long_term_filter`
- `model_id`, `api_key`, `api_url`, `max_tokens`, `temperature`

### Human In The Loop

HITL stages raise a resumable interrupt instead of reading from stdin. Enable HITL when constructing the stage:

```python
from IkaCore import IkaBaseAgent, IkaStage

review = IkaStage(
    name="review",
    prompt="Ask the user which filtering option to use, then finish the review.",
    tools=[],
    hitl=True,
)
agent = IkaBaseAgent(
    name="reviewer",
    description="Requests a filtering decision",
    prompt="Review the filtering options.",
    Stages=[review],
    model_id="gpt-4o-mini",
    api_key="...",
)
agent.enable_snapshots("state/snapshots.db")
result = agent.execution()

if result.get("status") == "awaiting_user_input":
    resumed = agent.resume_from_snapshot(
        result["snapshot_id"],
        resume_input="Use the stricter filtering option.",
    )
```

The interrupt payload includes the stage name, question, and snapshot identifier when snapshots are enabled.

### Memory

Set `memory=True` to enable memory tools, then initialize the corresponding backends. The flag alone does not create storage. Memory access is controlled with `memory_access`.

Supported access keys:

- `short_term_save`
- `short_term_search`
- `long_term_save`
- `long_term_search`

Long-term save now expects structured arguments, not a pipe-delimited string.

This self-contained example uses a synchronous in-process `StorageBackend` implementation
with literal substring matching. It needs no Mem0 dependency or credentials, and its data
lasts only for the lifetime of the backend instance.

```python
from datetime import datetime
from typing import Any

from IkaCore import IkaBaseAgent
from IkaMem import LTMemItem, LTMemory, STMemory


class LocalStorage:
    def __init__(self):
        self.rows = []

    def save(self, value: Any, metadata: dict[str, Any]) -> None:
        self.rows.append({"data": value, "metadata": dict(metadata)})

    def search(self, query: str, limit: int, score_threshold: float) -> list[Any]:
        # Exact substring matches have score 1.0 in this demonstration.
        return [row for row in self.rows if query.lower() in str(row["data"]).lower()
                and score_threshold <= 1.0][:limit]

    def reset(self) -> None:
        self.rows.clear()


memory_access = {
    "short_term_save": True,
    "short_term_search": True,
    "long_term_save": True,
    "long_term_search": True,
}

agent = IkaBaseAgent(
    name="memory_agent",
    description="Stores findings",
    prompt="Analyze and remember relevant results.",
    model_id="gpt-4o-mini",
    api_key="...",
    memory=True,
    memory_access=memory_access,
)
agent.short_term_memory = STMemory(storage=LocalStorage())
agent.short_term_memory.agent = agent.name
agent.long_term_memory = LTMemory(storage=LocalStorage())
agent.long_term_memory.agent = agent.name

agent.short_term_memory.save("The project uses Python", {"source": "README"})
assert agent.short_term_memory.search("Python")[0]["data"] == "The project uses Python"
agent.long_term_memory.save(LTMemItem(
    agent=agent.name,
    task="Inspect the project",
    expected_output="A Python framework",
    datetime=datetime.now().isoformat(),
    quality=1.0,
))
assert agent.long_term_memory.search("Inspect")[0]["metadata"]["expected_output"] == "A Python framework"
```

`STMemory` and `LTMemory` accept any backend implementing `save(value, metadata)`, `search(query, limit, score_threshold)` and `reset()`. For optional Mem0 storage, install/configure `mem0ai`, call `agent.init_short_term_memory({"provider": "mem0", "config": {...}})`, and assign `LTMemory(embedder_config=...)` for long-term storage. See the [memory setup guide](../src/IkaMem/docs/README.md).

### Provider Selection

Provider routing is derived from `model_id`, unless you override `api_url`.

Default behavior:

- OpenAI models use the Responses API by default.
- Set `use_responses_api=False` to target OpenAI Chat Completions.
- Anthropic, Gemini, DeepSeek, and OpenRouter URLs are inferred from `model_id`.
- Codex model IDs ending in `-codex`, such as `gpt-5.3-codex`, route to the Codex backend.
- If you pass an explicit non-OpenAI `api_url`, that URL is respected.

Example:

```python
agent = IkaBaseAgent(
    name="chat_agent",
    description="Uses OpenAI chat completions",
    prompt="Reply briefly.",
    model_id="gpt-4o-mini",
    api_key="...",
    use_responses_api=False,
)
```

#### Codex Auth

IkaCore supports the Codex backend at `https://chatgpt.com/backend-api/codex/responses`. Codex bills against the caller's ChatGPT plan and uses a bearer token, not an OpenAI API key.

The provider does not read local credentials automatically. `api_key` is treated as the literal bearer token, matching the other providers. You can source that bearer from an environment variable, a secret store, or the optional `codex_auth.get_bearer()` helper.

```python
import os

from IkaCore import IkaBaseAgent
from IkaModel.codex import CODEX_API_URL, codex_auth

bearer = os.getenv("CODEX_BEARER") or codex_auth.get_bearer()

agent = IkaBaseAgent(
    name="codex_agent",
    description="Uses the Codex backend",
    prompt="Reply briefly.",
    model_id="gpt-5.3-codex",
    api_key=bearer,
    api_url=CODEX_API_URL,
)
```

`codex_auth.get_bearer()` reads `~/.codex/auth.json`, or `$CODEX_HOME/auth.json` when `CODEX_HOME` is set. It refreshes the access token when it is close to expiry and writes the refreshed token back to `auth.json`. If `auth.json` does not exist, run `codex login` first.

Bare `gpt-5.x` model names do not auto-route to Codex because they overlap with standard OpenAI Responses models. For those, pass `api_url=CODEX_API_URL` explicitly. A Codex `401` means the bearer was rejected; refresh the token with `codex_auth.get_bearer(force_refresh=True)` and retry.

### Workflows

`IkaWorkflow` runs multi-agent graphs.

```python
from IkaCore import IkaBaseAgent, IkaWorkflow, WorkflowEdge, WorkflowNode

researcher = IkaBaseAgent(
    name="researcher",
    description="Collects context",
    prompt="Expand the topic and extract the important details.",
    model_id="gpt-4o-mini",
    api_key="...",
)

writer = IkaBaseAgent(
    name="writer",
    description="Summarizes context",
    prompt="Summarize the received context in a short paragraph.",
    model_id="gpt-4o-mini",
    api_key="...",
)

workflow = IkaWorkflow(
    name="two_step",
    description="Research then summarize",
    nodes=[
        WorkflowNode(name="research", agent=researcher),
        WorkflowNode(name="write", agent=writer),
    ],
    edges=[
        WorkflowEdge(source="research", target="write", edge_type="next"),
    ],
    start_node="research",
)

results = workflow.run(initial_context="The number 42 in popular culture.")
```

Current workflow semantics:

- `"next"` edges pass summary context downstream.
- `"child"` edges are tool-only. They wire a child agent into a specific parent stage with `stage_index`.
- `"child"` edges do not auto-execute as standalone workflow nodes.
- A `"child"` edge without `stage_index` is invalid.
- Sync and async workflow execution now use the same child-edge semantics.

### Snapshots

Using an agent configured above, opt in to durable run, stage, and tool state through one SQLite `SnapshotStore`:

```python
store = agent.enable_snapshots("state/snapshots.db")
result = agent.execution()
if result.get("status") in {"paused", "awaiting_user_input"}:
    resumed = agent.resume_from_snapshot(result["snapshot_id"], resume_input="Continue.")
```

A newly constructed, equivalently configured agent can enable the same database and
resume a saved ID. Snapshots are immutable. A boundary's snapshot, event, and explicit
pause status commit together; failed writes roll back. Connections are scoped to the
operation/transaction, and SQLite's default durability is retained.

V2 removes `CheckpointStore`, `load_checkpoint()`, constructor `checkpoint` and
`checkpoint_db_path` options, stage `checkpoint`, and `execution(checkpoint_uid=...)`.
Use `enable_snapshots()` and `resume_from_snapshot()` instead. Returned IDs are named
`snapshot_id`. Old checkpoint databases are not converted or deleted; resume them with
v1 before migrating. Snapshot replay does not restore a Python stack or undo tool effects.

### Logging And Debugging

Useful options:

- `logging_level=1..3`
- `logging_file="logs.txt"`
- `show_usage_level0=True`
- `IKA_DUMP_REQUESTS=/path/to/dir` to dump request/response payloads for debugging

Optional live provider smoke tests are available for checking real provider plumbing outside normal CI:

```bash
IKACORE_LIVE_PROVIDER_TESTS=1 OPENAI_API_KEY=... pytest src/IkaTest/test_live_provider_smoke.py
```

The live smoke suite supports OpenAI, Anthropic, DeepSeek, Gemini, and OpenRouter. When live smoke is enabled, at least one provider must be fully configured. For providers without stable project defaults, set the matching `IKACORE_LIVE_<PROVIDER>_MODEL` variable before running it.

For live Codex text, tool, agent, resume and cancellation checks, see
[the integration runner](../CONTRIBUTING.md#live-codex-integration-checks).

### Examples

The repo includes runnable examples in `examples/`:

- `example_fileagent.py`
- `example_stages.py`
- `example_hitl.py`
- `example_workflow.py`
- `example_workflow_advanced.py`

These examples assume the package is installed, for example with `pip install -e .`, and should not mutate `sys.path`.
