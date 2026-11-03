## IkaCore Architecture Notes

This document explains how the current IkaCore runtime is structured and where the main execution boundaries live.

### Package Layout

- `src/IkaCore`: agent runtime, stages, workflow graph, logging, checkpoints
- `src/IkaModel`: provider payload builders, transport layer, tool execution loop, summarization helpers
- `src/IkaMem`: short-term and long-term memory abstractions
- `src/IkaTest`: regression tests for runtime behavior

### Core Runtime Model

`IkaBaseAgent` is the public entry point. It mixes together four responsibilities:

- `AgentMemoryMixin`: memory initialization and memory tool helpers
- `AgentToolsMixin`: tool conversion, stage tool assembly, subagent delegation
- `AgentExecutionMixin`: thin execution entry points
- `AgentHelpersMixin`: shared runtime helpers, validation, checkpoint and HITL support

The constructor in [`agents.py`](../src/IkaCore/agents.py) is the contract that the rest of the repo now follows.

### Execution Modes

There are three relevant execution modes.

1. Simple agent execution
   The agent uses its top-level prompt and tool set. This is the narrowest loop.

2. Staged execution
   The agent walks through `IkaStage` objects. Each stage rebuilds the active prompt and tool set while preserving prior transcript context.

3. Workflow execution
   `IkaWorkflow` runs multiple agents and propagates summaries across graph edges.

### Prompt And Transcript Flow

The runtime keeps a normalized `message_history` with:

- `system`
- `first_input`
- `summary`
- `messages`

The important current rule is:

- the stage-specific prompt lives in `first_input`
- the long-lived transcript lives in `messages`
- summarization can compress prior transcript into `summary`

This avoids the older bug where stage 0's first prompt kept being replayed as the active user instruction for later stages.

### Control Tools

IkaCore models agent control through tool calls instead of hidden side channels.

Important control tools:

- `agent_end`
- `stage_end`
- `change_stage`
- `ask_user`

The control parser path is now unified. The dead duplicate control-call implementation was removed from the active execution path, so provider behavior is interpreted in one place.

### Subagents

Subagents are exposed as tools. The relevant code is in [`agent_tools.py`](../src/IkaCore/agent_tools.py).

Current boundary rules:

- parent agents may delegate work to subagents through a tool call
- delegated task input is treated as untrusted content
- the delegated task is not merged into the subagent's instruction prompt
- original subagent prompt state is restored after execution

That change closed the prompt-injection bug found during the security scan.

### Workflow Semantics

`IkaWorkflow` supports two edge types:

- `next`
- `child`

The semantics are intentionally strict:

- `next` means downstream dependency and summary propagation
- `child` means stage wiring only
- `child` requires `stage_index`
- `child` does not create an auto-executed child node

This behavior is enforced by tests in [`test_workflow_semantics.py`](../src/IkaTest/test_workflow_semantics.py).

### Provider Layer

`IkaModel` separates three concerns:

1. Provider detection and URL selection in [`request_interface.py`](../src/IkaModel/request_interface.py)
2. Provider-specific payload builders under `openai/`, `anthropic/`, `deepseek/`, `gemini/`, `openrouter/`
3. The transport and tool-call loop in [`chat_interface.py`](../src/IkaModel/chat_interface/chat_interface.py) and [`chat_runtime.py`](../src/IkaModel/chat_interface/chat_runtime.py)

Important current behavior:

- OpenAI Responses is the default OpenAI backend
- OpenAI Chat Completions requires `use_responses_api=False`
- Codex is a separate provider that targets `CODEX_API_URL`
- explicit non-OpenAI URLs are respected

Codex auth is intentionally not automatic in the request path. The Codex provider treats `BareBoneModel.api_key` as the literal bearer token. Users can pass their own bearer token, or call `IkaModel.codex.codex_auth.get_bearer()` to read and refresh the Codex CLI credentials from `~/.codex/auth.json` or `$CODEX_HOME/auth.json`.

### HITL And Resume

Human-in-the-loop no longer depends on blocking terminal input.

Current behavior:

- `ask_user` raises a structured interrupt
- the interrupt can be checkpointed
- `resume_execution(..., resume_input=...)` continues the run

Regression coverage lives in [`test_hitl_interrupt_resume.py`](../src/IkaTest/test_hitl_interrupt_resume.py).

### Checkpointing

Checkpoint persistence is implemented in [`checkpoint.py`](../src/IkaCore/checkpoint.py).

The current implementation is intentionally simple:

- SQLite-backed storage
- JSON payloads
- resumable run metadata for agent/stage/HITL state

It is materially better than the older coarse resume path, but it is still lighter-weight than full durable orchestration systems such as LangGraph.

### Memory

`IkaMem` exposes:

- `STMemory`
- `LTMemory`
- `Mem0Store`

Agent-side memory tools are assembled by `AgentMemoryMixin`. Long-term save and search now use structured tool schemas instead of the old string-splitting format.

### Maintenance Gates

The repo now has explicit gates for the maintenance risks that used to drift:

- no production function or class may grow to 70 lines or more
- broad `except Exception` boundaries are limited to the current intentional runtime isolation points
- `pyright` checks import/name, call, argument, assignment, return, optional, iterable, and attribute access regressions
- coverage must stay above the configured project threshold
- optional live provider smoke tests are kept separate from deterministic CI

The remaining maturity gap is not local unit behavior. It is broader integration confidence: real provider smoke coverage is opt-in, and long-running degraded-network workflows still need periodic manual or scheduled runs with credentials.
