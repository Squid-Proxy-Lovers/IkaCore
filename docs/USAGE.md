## IkaCore Usage Guide

### 1. Overview

IkaCore is an agent framework built around `IkaBaseAgent` plus modular mixins for tools, memory, execution, and stages. This guide shows how to configure agents, wire tools and memory, and run them in sync or async mode.

### 2. Core Concepts

- **`IkaBaseAgent` (`src/IkaCore/agents.py`)**: main abstraction you instantiate. Wraps:
  - Model configuration (provider, model id, API key, URL, temperature, max tokens)
  - Tools and optional subagents
  - Optional multi stage workflows (`Stages`)
  - Optional memory (short term and long term)
  - Optional output validation (`final_answer_checks`)
  - Optional async execution (`use_async`)

- **Models (`src/IkaModel`)**
  - `BareBoneModel` in `base.py` is the minimal model descriptor.
  - Provider specific payload builders: `deepseek.py`, `openai.py`, `claude.py`, `google.py`.
  - `chat_interface.py` provides:
    - `chat(...)`, `execute_tool_calls(...)` (sync)
    - `async_chat(...)`, `async_execute_tool_calls(...)` (async)
    - `summarise_message_history(...)` and `async_summarise_message_history(...)`.

- **Tools (`IkaCore.tools` + `src/Adapter/adapter.py`)**
  - Tools are represented as `IkaTools` and `AgentTool`.
  - The adapter converts normal Python functions into tools and can create delegate tools that call other agents.

- **Stages (`src/IkaCore/stages.py`)**
  - `IkaStage` describes one step in a workflow: prompt, tools, step limits, HITL, memory access, and allowed back edges.

- **Memory (`src/IkaMem`)**
  - Short term memory: per run, for local context.
  - Long term memory: cross run, backed by mem0 storage when configured.
  - `AgentMemoryMixin` provides helper methods.

### 3. Installation and API Keys

From the repo root:

```bash
cd /Users/tarun/IkaCore
pip install -r requirements.txt  # if you maintain one
```

API keys are typically loaded from an `apikeys` file (see `src/IkaModel/test_base.py`):

```text
deepseek=YOUR_DEEPSEEK_KEY
openai=YOUR_OPENAI_KEY
claude=YOUR_ANTHROPIC_KEY
gemini=YOUR_GEMINI_KEY
```

Your scripts (for example `Example/agents/run_recon_only.py`) choose the right key based on the selected model id.

### 4. IkaBaseAgent Configuration

Constructor (simplified):

```python
from IkaCore.agents import IkaBaseAgent

agent = IkaBaseAgent(
    name="my_agent",
    description="Does X",
    prompt="User task prompt here",
    system_prompt=None,
    start_prompt=None,
    end_prompt=None,
    role="worker",
    tools=my_tools,                 # Optional[List[IkaTools]]
    model_id="deepseek-chat",
    api_key="...",
    api_url=None,                   # auto derived from model_id if None
    max_tokens=20000,
    temperature=0.0,
    checkpoint=False,
    Batch=False,
    BatchMax=3,
    Stages=None,                    # or List[IkaStage]
    subagents=None,                 # alternative to Stages / next_agent
    next_agent=None,
    feedback_agent=None,
    maxsteps=10,
    step_timeout=900,
    rate_limit_per_min=None,
    per_tool_rate_limit=None,
    memory=False,
    memory_access=None,
    final_answer_check=None,        # list[callable] returning bool
    logging_level=0,
    logging_file="logs.txt",
    show_usage_level0=True,
    checkpoint_db_path="checkpoints.db",
    enable_summarization=True,
    use_async=False,                # True to use async_chat backend
)
```

Notes:

- Required fields: `name`, `description`, `prompt`, `role`, `model_id`, `api_key`. Missing any causes a `ValueError`.
- If `Stages` is set you must not set `subagents`, `next_agent` or `feedback_agent`.
- Without `Stages` you can have either `subagents` or `next_agent`, not both.

### 5. Running Agents

#### 5.1 Simple agents (no stages)

When `Stages` is not set, the agent uses `run_simple()` internally:

```python
result = agent.execution()
print(result["final_message"])
print(result["summary"])
```

Flow:

- Builds `AgentTool` list from `self.tools` plus memory tools and an `agent_end` control tool.
- Creates a `BareBoneModel` with `get_barebone`.
- Steps up to `maxsteps` using:
  - `chat(...)` when `use_async=False`.
  - `async_chat(...)` (via `asyncio.run`) when `use_async=True`.
- Uses `parse_control_calls(...)` to detect `agent_end` and extract the final response.
- Wraps the result via `_build_final_output(...)`, which may also summarise the conversation.

#### 5.2 Staged agents

Create one or more `IkaStage` instances and pass them as `Stages`:

```python
from IkaCore.stages import IkaStage

stage = IkaStage(
    name="analysis_stage",
    prompt="Perform detailed analysis of the input.",
    tools=my_stage_tools,
    stage_max_step=5,
    hitl=False,
    memory_access={"short_term_save": True, "short_term_search": True},
    allowed_back_to=[],
)

agent = IkaBaseAgent(
    name="analysis_agent",
    description="Multi-stage analysis agent",
    prompt="Base prompt",
    role="analyst",
    tools=[],
    model_id="deepseek-chat",
    api_key="...",
    Stages=[stage],
    maxsteps=20,
    memory=True,
    use_async=True,
)

result = agent.execution()
```

Flow:

- `execution()` iterates stages using `execute_stage(stage_index, remaining_steps)`.
- Each `execute_stage`:
  - Builds:
    - Stage tools from `stage.tools`.
    - Subagent tools from any `stage.subagents`.
    - Stage specific memory tools based on `stage.memory_access` (falls back to agent level mask).
  - Calls `chat` or `async_chat` per `use_async`.
  - Processes control tools via `parse_control_calls`:
    - `agent_end`: stop the entire agent and return a final result.
    - `stage_end`: go to next stage in order.
    - `change_stage`: jump to an earlier allowed stage index in `stage.allowed_back_to`.
  - Optionally prompts the user for additional input on HITL stages (`stage.hitl=True`).

Common `IkaStage` fields you can use:

- `name`: human readable stage name, used in logs.
- `prompt`: stage specific prompt; if empty, the agent’s main prompt is used.
- `tools`: list of tools this stage can call.
- `stage_max_step`: max tool/model steps for this stage (capped by agent `maxsteps`).
- `hitl`: if `True`, the stage can pause and collect user input between steps.
- `memory_access`: per stage override for memory tool access.
- `long_term_filter`: optional filter applied when reading long term memory.
- `subagents`: optional subagents that are converted into tools for this stage.
- `allowed_back_to`: list of stage indices the agent is allowed to jump back to via `change_stage`.
- Per-stage model overrides (when unset, the agent's values are used):
  - `model_id`: model to use for this stage (e.g. `gpt-4o`, `claude-sonnet-4`). If set and `api_url` is not, `api_url` is derived from `model_id`.
  - `api_key`: API key for this stage (required when switching provider, e.g. OpenAI to Anthropic).
  - `api_url`: optional custom API base URL.
  - `max_tokens`, `temperature`: overrides for this stage.

#### 5.3 Chaining agents

When not using `Stages`:

- `subagents`: turned into tools and called from the parent agent like any other tool.
- `next_agent`: after the current agent finishes:
  - Its summary is placed into `next_agent.message_history["first_input"]["message"]`.
  - Then `next_agent.execution()` is run.

This is useful for linear multi agent flows.

### 6. Workflows (IkaWorkflow)

For multi agent flows you can use `IkaWorkflow` (`src/IkaCore/workflow.py`) to define a small graph of agents and their relationships.

#### 6.1 Workflow building blocks

- **`WorkflowNode`**
  - `name`: node id in the graph.
  - `agent`: an `IkaBaseAgent` instance.
  - `stage_wiring`: optional mapping `stage_index -> {"subagents": [IkaBaseAgent, ...]}` used to inject subagents into specific stages.
  - `instances`: how many parallel instances of the agent to run.
  - `instance_inputs`: optional list of per instance prompts/inputs.

- **`WorkflowEdge`**
  - `source`, `target`: node names.
  - `edge_type`:
    - `"next"`: linear or branching progression; target runs after source and receives its summary as context.
    - `"child"`: parent/child relationship; child runs as a helper and returns its summary back to the parent.
  - `stage_index` (optional): if set on a `"child"` edge, the child agent is wired into that parent stage as a subagent.

- **`IkaWorkflow`**
  - Arguments:
    - `name`, `description`.
    - `nodes`: list of `WorkflowNode`.
    - `edges`: list of `WorkflowEdge`.
    - `compress_hook`: optional function `(List[str], IkaBaseAgent) -> str` to compress upstream summaries into a single context string.
    - `start_node`: optional name of the first node; defaults to the first node in `nodes`.
    - `async_executor`: optional `AsyncWorkflowExecutor`, otherwise a default is created.
    - `max_parallel_workers`: max parallel threads used by the async executor.

#### 6.2 How execution works

There are two entry points:

- **`run(initial_context: str | None = None, use_async: bool = False)`**
  - When `use_async=False`:
    - Calls internal `_run_node(start_node, upstream_contexts)`.
    - For each node:
      - Prepares context by compressing upstream summaries using `compress_hook`.
      - Applies stage wiring so staged agents see their child agents as subagents.
      - Calls `node.agent.execution()` and captures `final_message` + `summary`.
      - For each `"child"` edge, immediately runs the child node and records its summary under `child_summaries`.
      - For each `"next"` edge, passes a compressed combination of parent summary and child summaries downstream.
    - Returns a dict of `node_name -> WorkflowResult`.

  - When `use_async=True`:
    - Delegates to `run_async(initial_context)`, which uses `AsyncWorkflowExecutor` to run nodes in parallel where dependencies allow.

- **`run_async(initial_context: str | None = None)`**
  - Maintains dependency sets so only nodes whose upstream `"next"` dependencies are complete can start.
  - For each ready node:
    - Optionally creates multiple instances (per `instances` / `instance_inputs`).
    - Schedules them on `AsyncWorkflowExecutor.schedule_node(...)`, which:
      - Runs each agent in a thread using `agent.execution()` or `agent.async_execution()` if present.
    - Aggregates results:
      - First instance becomes the primary `WorkflowResult`.
      - Additional instances append their summaries to the primary summary.
  - Updates upstream contexts for downstream `"next"` edges using node summaries.
  - Ends by returning a dict of `node_name -> WorkflowResult`.

#### 6.3 Example workflow usage

Minimal example wiring two agents A -> B:

```python
from IkaCore.workflow import IkaWorkflow, WorkflowNode, WorkflowEdge

node_a = WorkflowNode(name="analysis", agent=analysis_agent)
node_b = WorkflowNode(name="report", agent=report_agent)

edge = WorkflowEdge(source="analysis", target="report", edge_type="next")

workflow = IkaWorkflow(
    name="analysis_pipeline",
    description="Two step pipeline: analysis then reporting",
    nodes=[node_a, node_b],
    edges=[edge],
)

results = workflow.run(initial_context="User request here", use_async=False)
print(results["report"].final)
print(results["report"].summary)
```

To wire a child helper agent into stage 0 of a staged parent:

```python
parent_node = WorkflowNode(name="parent", agent=parent_agent)
child_node = WorkflowNode(name="helper", agent=helper_agent)

edge = WorkflowEdge(source="parent", target="helper", edge_type="child", stage_index=0)

workflow = IkaWorkflow(
    name="parent_with_helper",
    description="Parent staged agent that can call helper as a subagent",
    nodes=[parent_node, child_node],
    edges=[edge],
)
results = workflow.run()
```

The `child` edge with `stage_index=0` means:

- `parent_agent` will receive `helper_agent` as a subagent in stage 0.
- The helper’s summary is injected back into the parent and included in downstream context.

### 7. Tools

#### 6.1 Converting functions to tools

Use the adapter patterns from `src/Adapter/adapter.py` and `test_adapter.py`:

```python
from IkaCore.tools import tool
from Adapter.adapter import function_to_ika_tool

@tool
def add_numbers(a: int, b: int) -> int:
    return a + b

ika_tool = function_to_ika_tool(add_numbers)
agent = IkaBaseAgent(..., tools=[ika_tool])
```

`function_to_ika_tool`:

- Reads name and docstring.
- Builds JSON schema for parameters from type hints.
- Produces an `IkaTools` object usable by `IkaBaseAgent`.

#### 6.2 Delegate tools to sub-agents

From `Adapter/adapter.py`:

```python
from Adapter.adapter import make_delegate_tool

delegate_tool = make_delegate_tool("sub_agent", sub_agent_instance, "Delegate to sub-agent")
parent_agent = IkaBaseAgent(..., tools=[delegate_tool])
```

The model can then call `sub_agent` as a tool; the delegate forwards the task into the sub agent and returns its final result.

### 8. Memory

Enable memory at agent level:

```python
agent = IkaBaseAgent(
    ...,
    memory=True,
    memory_access={
        "short_term_save": True,
        "short_term_search": True,
        "long_term_save": False,
        "long_term_search": False,
    },
)
```

When memory is enabled:

- Short term tools:
  - `short_term_save(input)`
  - `short_term_search({query, limit, score_threshold})`
- Long term tools:
  - `long_term_save(input)` with `"task|output"` format.
  - `long_term_search({query, limit, score_threshold})`

Stage specific `memory_access` overrides the agent level mask, so you can allow memory only for certain stages.

### 9. Async vs Sync Execution

Set `use_async=True` on the agent to opt into the async pipeline:

- `execute_stage`:
  - uses `async_chat` via `asyncio.run`.
- `run_simple`:
  - also uses `async_chat` when `use_async=True`.

Async benefits:

- Tools run via `async_execute_tool` and `async_execute_tool_calls`, which can:
  - Await coroutine tools directly.
  - Run sync tools in a thread pool without blocking the event loop.
  - Execute parallel tools concurrently when configured.

If `use_async=False`, everything runs through the original synchronous `chat` and `execute_tool_calls`.

### 10. Control Tools and Final Outputs

Control tools:

- `agent_end`: signals final completion for the agent.
- `stage_end`: moves to the next stage.
- `change_stage`: jumps back to an earlier allowed stage.

`agent_end` contract:

- The model must provide the final answer in `input`, `final`, or `message` arguments.
- The agent:
  - Validates that the content is not empty, `"{}"`, `"[]"`, `"null"`, etc.
  - Tries to use JSON content if present.
  - Falls back to extracted JSON or raw model content via `_fallback_final_content` if arguments are weak.

`execution()` always returns a dict:

```python
{
    "final_message": "<string>",
    "summary": "<string>",  # may be empty if summarisation is disabled
}
```

### 11. Example: Recon Agent

Example runner: `Example/agents/run_recon_only.py`.

High level steps:

1. Parse CLI args (subnet, model, max steps, keys file).
2. Load API keys and choose the key for `args.model`.
3. Build recon tools via `get_recon_toolset()`.
4. Create an `IkaStage` with recon prompt and tools.
5. Instantiate `IkaBaseAgent`:

```python
recon_agent = IkaBaseAgent(
    name="recon_agent",
    description="Network reconnaissance agent",
    prompt="You are a network reconnaissance agent. Perform comprehensive network analysis.",
    role="recon",
    tools=[],
    model_id=args.model,
    api_key=api_key,
    maxsteps=args.max_steps,
    Stages=[recon_stage],
    enable_summarization=True,
    logging_level=1,
    show_usage_level0=True,
    use_async=True,
)
```

6. Set `recon_agent.message_history["first_input"]["message"]` to a task prompt that contains subnet, objectives, and output file location.
7. Call `recon_agent.execution()` and consume `final_message` plus any JSON/network artifacts written by the tools.

This pattern can be copied to build other specialized agents with stages, tools, and memory. 

