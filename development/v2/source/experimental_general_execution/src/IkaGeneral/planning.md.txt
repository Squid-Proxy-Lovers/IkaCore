# IkaGeneral: Continuous General Agent Execution System

## Context

IkaCore provides a mature agent framework with multi-provider LLM support, tool execution, stage-based workflows, memory, and context management. **IkaGeneral** builds a higher-level execution environment on top of this — a continuous, autonomous agent execution system with explicit control over its own context via context packs, the ability to spawn and manage recursive sub-agents, dynamically create tools at runtime, and navigate its own execution history via checkpoints and branching.

**Existing IkaCore infrastructure used:**

- `IkaBaseAgent` (src/IkaCore/agents.py) — base agent with mixins
- `IkaTools` (src/IkaCore/tools.py) — tool definitions with execute callbacks
- 
- `IkaStage` (src/IkaCore/stages.py) — stage-based execution
- `BareBoneModel` (src/IkaModel/base.py) — multi-provider LLM abstraction
- `chat()` / `async_chat()` (src/IkaModel/chat_interface/chat_interface.py) — chat loop with context management
- Summarization/compaction (src/IkaModel/summarization.py) — context overflow handling
- `CheckpointStore` (src/IkaCore/checkpoint.py) — SQLite checkpoint persistence

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  IkaGeneral                          │
│                                                     │
│  ┌─────────────────────────────────────────────┐    │
│  │         IkaExecutionEnvironment              │    │
│  │                                              │    │
│  │  ┌──────────┐  ┌───────────┐  ┌──────────┐  │    │
│  │  │ Context  │  │ Execution │  │ Checkpoint│  │    │
│  │  │ Pack     │  │ Memory    │  │ Manager   │  │    │
│  │  │ Registry │  │           │  │           │  │    │
│  │  │ (global) │  │ (per-env) │  │ (auto+   │  │    │
│  │  │          │  │           │  │  named)   │  │    │
│  │  └──────────┘  └───────────┘  └──────────┘  │    │
│  │                                              │    │
│  │  ┌──────────┐  ┌───────────┐  ┌──────────┐  │    │
│  │  │ Tool     │  │ SubAgent  │  │ Branch   │  │    │
│  │  │ Registry │  │ Registry  │  │ Manager  │  │    │
│  │  │ (global) │  │ (global)  │  │          │  │    │
│  │  └──────────┘  └───────────┘  └──────────┘  │    │
│  │                                              │    │
│  │  ┌──────────┐  ┌───────────┐                 │    │
│  │  │ Cost     │  │ Introspect│                 │    │
│  │  │ Tracker  │  │ ion Tools │                 │    │
│  │  └──────────┘  └───────────┘                 │    │
│  └──────────────────────────────────────────────┘    │
│                                                     │
│  Uses: IkaBaseAgent, IkaTools, BareBoneModel,       │
│        chat/async_chat, summarization               │
└─────────────────────────────────────────────────────┘
```

---

## Design Decisions

### 1. Execution Model — Continuous, Agent-Driven with User Interruption

The agent runs **continuously and autonomously**, deciding when to use tools, branch, reset, spawn sub-agents, or terminate. The user can **interrupt at any point** via signal + message — the current step is paused (if possible), then the user's message is injected into execution memory.

### 2. Dual-Layer Context System

**Execution Memory** (per-environment):

- Running record of tool calls + results
- Required by model providers (tool_call + result pairs)
- Gets checkpointed/reset
- Auto-compacted via IkaCore summarization on context overflow

**Context Packs / Memory Vars** (global, shared):

- In-memory, mutable, named containers
- Hybrid format: free-form content + metadata (name, source, creation time, tags)
- Persist across checkpoint resets — never destroyed by rollbacks
- **Shared references** between parent and sub-agents (updates visible to both)
- Protected by async locks for concurrent access

**Context pack injection modes:**

- **Pre-loaded for sub-agents**: Pass packs at sub-agent creation time
- **Explicit load by agent**: Agent calls `load_context_pack`. Never auto-loaded
- **Chunked loading**: If a pack exceeds remaining context window, load as much as fits with a continuation token. Agent loads next chunk when needed

**Loaded pack placement in message array:**

```
[system_prompt] → [initial_prompt] → [loaded_context_packs] → [execution_memory...]
```

Loaded packs appear as user messages **after the system prompt and initial prompt, before execution memory**.

### 3. Branching — N Prompts, Parallel Exploration

`branch_execution_path(n, prompts)` — Agent provides **N different task/prompt strings**. Each branch starts a **fresh sub-environment** with that prompt + current loaded context packs (shared references). Branches run as asyncio tasks, **isolated in execution memory**, sharing mutable context packs with lock protection.

When all branches complete, their `end_execution(result)` outputs are returned to the **parent agent** as a structured list for LLM-driven merge/selection.

**Constraint**: Can only branch when **no sub-agents are currently running**.

### 4. Sub-Agents — Fully Recursive, Configurable Model

- Own full execution environment (context packs, branching, checkpointing, sub-sub-agents)
- No depth limits
- Agent specifies **any supported model + parameters** (temperature, max tokens) per sub-agent
- **Final result only** — parent calls `run_subagent`, waits, gets result after sub-agent's `end_execution`
- Pre-loaded context packs are **shared references** (sub-agent updates visible to parent)

### 5. Edit Semantics — Modify Definitions for Future Use

`edit_tool_call` / `edit_subagent` modify definitions for **future invocations** only. Definitions are **GLOBAL objects** — persist across checkpoint resets.

### 6. Architecture — Standalone on Top of IkaCore

Separate module (`src/IkaGeneral/`) importing IkaCore components. Own execution loop. Does not modify IkaCore internals.

### 7. Checkpointing — Automatic + Agent-Created

- Auto-checkpoint every N steps
- Agent can create named checkpoints via `create_checkpoint(name)`
- Reset targets by name or step number

**Checkpoints capture**: execution memory only.
**Global (persist across resets)**: context packs, sub-agent defs, tool defs, loaded-pack tracking.

### 8. Introspection — Full Suite

`list_tools`, `list_subagents`, `list_context_packs`, `view_execution_history`, `view_checkpoints`, `check_budget`

### 9. Error Handling

- Branch/sub-agent failure → error result returned to parent agent for decision
- Context overflow → auto-compact via IkaCore summarization

### 10. Dynamic Tool Creation — Agent-Written Python, Trusted

`make_tool_call(name, description, parameters, code)` — agent writes Python code for a tool's execute function. Code runs directly in the main process. No sandbox. Agent is trusted.

### 11. Concurrency — Asyncio + Locks

Parallel branches as asyncio tasks. Shared context packs protected by async locks.

### 12. Configuration — Builder Pattern

```python
env = (IkaExecutionEnv.builder()
    .model("claude-sonnet-4-6")
    .system_prompt("You are a research agent...")
    .initial_prompt("Research X and produce a report...")
    .tools([web_search, file_read, ...])
    .context_packs([background_pack, data_pack])
    .cost_budget(5.00)
    .auto_checkpoint_interval(10)  # every 10 steps
    .build())

result = await env.run()
```

### 13. Guardrails — Cost Budget + Long-Running Tool Alerts

- **Cost budget**: Configurable dollar limit. Agent can introspect remaining budget via `check_budget()`. Execution stops when exhausted.
- **Long-running tool alert**: If a tool runs > 10 minutes, system notifies the agent and asks it to confirm the behavior is expected. Tool continues running in background — it is NOT killed.
- No step limits, depth limits, or branch limits.

---

## Complete Meta-Tool Set

### Context Pack Tools

- `create_context_pack(name, content, tags)` — Create a new named pack
- `load_context_pack(name, chunk?)` — Load into context window (chunked if needed)
- `unload_context_pack(name)` — Remove from active context
- `update_context_pack(name, content?, tags?)` — Mutate an existing pack
- `list_context_packs()` — List all packs with metadata, loaded status, size

### Sub-Agent Tools

- `make_subagent(name, description, prompt, tools, model?, model_params?, context_packs?)` — Create definition
- `run_subagent(name)` — Execute sub-agent, wait for final result
- `edit_subagent(name, ...)` — Modify definition for future runs
- `list_subagents()` — View all definitions and status

### Dynamic Tool Tools

- `make_tool_call(name, description, parameters, code)` — Create tool with agent-written Python
- `run_tool_call(name, args)` — Execute a registered tool
- `edit_tool_call(name, ...)` — Modify definition for future calls
- `list_tools()` — View all available tools

### Execution Control Tools

- `branch_execution_path(n, prompts)` — Fork into N parallel paths with distinct prompts
- `reset_to_checkpoint(target)` — Rollback execution memory (by name or step number)
- `create_checkpoint(name)` — Named checkpoint at current step
- `view_checkpoints()` — List available checkpoints (auto + named)
- `view_execution_history(limit?)` — Review past steps and results
- `check_budget()` — View remaining cost budget and usage
- `end_execution(result)` — Terminate with final result

---

## Key Data Structures

### ContextPack

```python
@dataclass
class ContextPack:
    name: str
    content: str          # free-form text, code, data
    tags: list[str]       # searchable tags
    source: str           # where this pack came from
    created_at: datetime
    updated_at: datetime
    size_tokens: int      # estimated token count
```

### ExecutionCheckpoint

```python
@dataclass
class ExecutionCheckpoint:
    name: str | None        # None for auto-checkpoints
    step_number: int
    execution_memory: list  # snapshot of message history
    loaded_packs: list[str] # which packs were loaded at this point
    created_at: datetime
```

### SubAgentDefinition

```python
@dataclass
class SubAgentDefinition:
    name: str
    description: str        # system prompt for the sub-agent
    prompt: str             # initial task prompt
    tools: list[str]        # tool names available to this sub-agent
    model: str | None       # None = inherit parent model
    model_params: dict      # temperature, max_tokens, etc.
    context_packs: list[str]  # pack names to pre-load
    status: str             # created, running, completed, failed
    last_result: Any        # result from most recent run
```

### DynamicToolDefinition

```python
@dataclass
class DynamicToolDefinition:
    name: str
    description: str
    parameters: dict        # JSON schema for tool parameters
    code: str               # Python code string for execute function
    created_at: datetime
```

---

## Execution Flow

1. **Initialization**: Builder creates `IkaExecutionEnvironment` with system prompt, initial prompt, tools, context packs, budget
2. **Context Assembly**: System prompt → initial prompt → loaded context packs (as user messages) → empty execution memory
3. **LLM Call**: Send assembled context to model, receive response with tool calls
4. **Tool Execution**: Execute requested tools (meta-tools or user-provided tools)
5. **Execution Memory Update**: Append tool calls + results to execution memory
6. **Auto-Checkpoint**: If step count hits interval, snapshot execution memory
7. **Cost Tracking**: Update budget usage, stop if exhausted
8. **Long-Running Check**: If any tool > 10 min, inject notification asking agent to confirm
9. **Context Management**: If approaching context limit, auto-compact via IkaCore
10. **User Interruption**: If signal received, pause and inject user message
11. **Loop**: Back to step 3 until `end_execution` is called

**Branching flow**: At step 4, if agent calls `branch_execution_path`, create N sub-environments, run them concurrently, collect results, inject results into parent's execution memory, continue at step 3.

**Sub-agent flow**: At step 4, if agent calls `run_subagent`, create sub-environment from definition, run to completion, return result to parent.