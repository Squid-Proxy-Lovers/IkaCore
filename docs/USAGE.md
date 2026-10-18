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
- `prompt`
- `model_id`
- `api_key`

`description` is optional and defaults to the name. Keep task text in `prompt`.
The advanced options are keyword-only frozen configuration values:

```python
from IkaCore import ExecutionLimits, ExecutionPolicy, OutputConfig, PromptConfig, ProviderConfig

observed_events = []
agent = IkaBaseAgent("configured", prompt="Answer clearly.", model_id="gpt-4o-mini", api_key="...",
                     provider=ProviderConfig(temperature=0.2),
                     prompts=PromptConfig(system_prompt="Be concise."),
                     limits=ExecutionLimits(maxsteps=10, max_tool_calls=5),
                     policy=ExecutionPolicy(max_step_extensions=0, max_stage_extensions=0),
                     output=OutputConfig(cli=False, subscribers=(observed_events.append,)))
```

`ExecutionPolicy(final_answer_checks=(check,))` preserves final-answer validation;
`validation_retries=3` and `validation_extra_steps=10` are explicit defaults.
Checks require a docstring and a boolean return value. Extension counts and sizes
also belong to this policy. `policy.use_async` selects the async request bridge.
Configuration is retained in `agent.config`; transcripts, counters, usage and the
current (possibly extended) budget live in `agent.state`.

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
Resume consumes the saved transcript once and appends the human answer before continuing.
Consumed steps, step limits, extension counts, and the remaining overall stage budget
survive snapshot resume, including when a new agent opens the database. Resuming partway
through a stage does not reset its allowance or charge prior steps twice to later stages.

For a snapshot-enabled workflow, read `info = workflow.snapshot_runtime_info()`, then
answer with `workflow.resume_from_snapshot(info["last_snapshot_id"], resume_input="YES")`.

A string answers the sole interrupted instance. When several instances need answers,
use a mapping such as `resume_input={"review:0": "YES", "review:1": "Revise"}`.
`info["interrupted_instances"]` lists those keys and their child snapshot IDs. Omitted
instances keep waiting; resuming without an answer does not repeat their model call.
You may also resume a child separately using its snapshot ID, then resume the workflow
without an answer: it reuses the completed child's output and continues dependents.

### Memory

Create backends explicitly and pass `MemoryTools(...).tools(access=...)` through the ordinary `tools` argument (or a stage’s tools). Core agents do not initialize or share memory. Filters belong to `MemoryTools.tools(long_term_filter=...)`.

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
from IkaMem.agent_tools import MemoryTools


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
)
short_term = STMemory(storage=LocalStorage())
short_term.agent = agent.name
long_term = LTMemory(storage=LocalStorage())
long_term.agent = agent.name

short_term.save("The project uses Python", {"source": "README"})
assert short_term.search("Python")[0]["data"] == "The project uses Python"
long_term.save(LTMemItem(
    agent=agent.name,
    task="Inspect the project",
    expected_output="A Python framework",
    datetime=datetime.now().isoformat(),
    quality=1.0,
))
assert long_term.search("Inspect")[0]["metadata"]["expected_output"] == "A Python framework"
memory_tools = MemoryTools(agent.name, short_term=short_term, long_term=long_term)
agent.tools.extend(memory_tools.tools(access=memory_access))
result = agent.execution()
assert result["final_message"]
```

`STMemory` and `LTMemory` accept any backend implementing `save(value, metadata)`, `search(query, limit, score_threshold)` and `reset()`. For optional Mem0 storage, install/configure `mem0ai`, construct `STMemory(embedder_config={"provider": "mem0", "config": {...}})` and `LTMemory(embedder_config=...)`, then pass them explicitly to `MemoryTools`. See the [memory setup guide](../src/IkaMem/docs/README.md).

### Provider Selection

Provider routing is derived from `model_id`, unless you override `api_url`.

Default behavior:

- OpenAI models use the Responses API by default.
- Set `provider=ProviderConfig(use_responses_api=False)` to target OpenAI Chat Completions.
- Anthropic, Gemini, DeepSeek, and OpenRouter URLs are inferred from `model_id`.
- Codex model IDs ending in `-codex`, such as `gpt-5.3-codex`, route to the Codex backend.
- If you pass an explicit non-OpenAI `api_url`, that URL is respected.

Example:

```python
from IkaCore import ProviderConfig
agent = IkaBaseAgent(
    name="chat_agent",
    description="Uses OpenAI chat completions",
    prompt="Reply briefly.",
    model_id="gpt-4o-mini",
    api_key="...",
    provider=ProviderConfig(use_responses_api=False),
)
```

#### Codex Auth

IkaCore supports the Codex backend at `https://chatgpt.com/backend-api/codex/responses`. Codex bills against the caller's ChatGPT plan and uses a bearer token, not an OpenAI API key.

The provider does not read local credentials automatically. `api_key` is treated as the literal bearer token, matching the other providers. You can source that bearer from an environment variable, a secret store, or the optional `codex_auth.get_bearer()` helper.

```python
from IkaCore import ProviderConfig
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
    provider=ProviderConfig(api_url=CODEX_API_URL),
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
- With workflow snapshots enabled, an instance returning `paused` or `awaiting_user_input` pauses the workflow and blocks its dependents in both execution modes. Interrupted outputs are not completed results.

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
Snapshot setup failures close their SQLite connection. Nested agents record snapshots
under their own frames and restore the parent's execution context when they return.

Historical agent resume checks tool effects recorded after the selected snapshot,
including nested calls. If resuming would repeat unsafe effects, explicit approval is
required: `agent.resume_from_snapshot(snapshot_id, allow_unsafe_replay=True)`.
Mid-tool replay also requires this approval. Completed snapshots return saved output;
post-model resume consumes the saved response without counting usage again. Fresh
executions clear replay state, including after a failed resume.

Enable workflow persistence with `workflow.enable_snapshots(path, control)` and inspect
`workflow.snapshot_runtime_info()` for `paused` and `last_snapshot_id`. Workflow
`resume_from_snapshot(snapshot_id, resume_input=None, allow_unsafe_replay=False)` reuses
completed instances and resumes saved interrupted children. Selective replay and historically cached
interruptions invalidate downstream results; nested or unfinished unsafe tool effects
require replay approval. See the HITL section for answering one or several instances.

V2 removes `CheckpointStore`, `load_checkpoint()`, constructor `checkpoint` and
`checkpoint_db_path` options, stage `checkpoint`, and `execution(checkpoint_uid=...)`.
Use `enable_snapshots()` and `resume_from_snapshot()` instead. Returned IDs are named
`snapshot_id`. Old checkpoint databases are not converted or deleted; resume them with
v1 before migrating. Snapshot replay does not restore a Python stack or undo tool effects.

### Logging And Debugging

Useful options:

- `output=OutputConfig(logging_level=1..3)`
- `OutputConfig(logging_file="logs.txt")`
- `OutputConfig(show_usage_level0=True)`
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

### Structured output events

Default CLI boxes retain the existing presentation. Disable them with
`output=OutputConfig(cli=False)`. Subscribe with `OutputConfig(subscribers=(callback,))`;
callbacks receive `RuntimeEvent(kind, data, instance_id)` with structured fields.
Subscriber exceptions propagate. Runtime scopes are restored on exceptions, follow
async tasks and tool workers, and isolate nested agents. Workflow output accepts the
same `output=` configuration and shares CLI buffering across parallel instances.

`IkaCore.output_adapters` ships `CLIAdapter`, `FileLogAdapter`, and opt-in `UsageAdapter`.
`logging_level` / `logging_file` select the file adapter. No file adapter is created
at level zero. Legacy `cli_output.set_stdout_enabled` / `set_output_sink` affect
only direct legacy CLI users; runtime presentation uses explicit configuration.
For standalone BareBoneModel/chat/tool calls use `with output_scope(RuntimeOutput(...))`.
An empty `RuntimeOutput()` emits no presentation and skips event allocation.
