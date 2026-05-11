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

The constructor in [agents.py](/Users/tarun/Malware-analysis-website/IkaCore/src/IkaCore/agents.py:31) is the contract that the rest of the repo now follows.

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

Subagents are exposed as tools. The relevant code is in [agent_tools.py](/Users/tarun/Malware-analysis-website/IkaCore/src/IkaCore/agent_tools.py:102).

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

This behavior is enforced by tests in [test_workflow_semantics.py](/Users/tarun/Malware-analysis-website/IkaCore/src/IkaTest/test_workflow_semantics.py:1).

### Provider Layer

`IkaModel` separates three concerns:

1. Provider detection and URL selection in [request_interface.py](/Users/tarun/Malware-analysis-website/IkaCore/src/IkaModel/request_interface.py:1)
2. Provider-specific payload builders under `openai/`, `anthropic/`, `deepseek/`, `gemini/`, `openrouter/`
3. The transport and tool-call loop in [chat_interface.py](/Users/tarun/Malware-analysis-website/IkaCore/src/IkaModel/chat_interface/chat_interface.py:1)

Important current behavior:

- OpenAI Responses is the default OpenAI backend
- OpenAI Chat Completions requires `use_responses_api=False`
- explicit non-OpenAI URLs are respected

### HITL And Resume

Human-in-the-loop no longer depends on blocking terminal input.

Current behavior:

- `ask_user` raises a structured interrupt
- the interrupt can be checkpointed
- `resume_execution(..., resume_input=...)` continues the run

Regression coverage lives in [test_hitl_interrupt_resume.py](/Users/tarun/Malware-analysis-website/IkaCore/src/IkaTest/test_hitl_interrupt_resume.py:1).

### Checkpointing

Checkpoint persistence is implemented in [checkpoint.py](/Users/tarun/Malware-analysis-website/IkaCore/src/IkaCore/checkpoint.py:1).

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

### Known Hygiene Gaps

The repo is cleaner than it was at the start of this work, but a few design debts remain:

- some runtime modules still mutate `sys.path` during import
- examples are still source-checkout scripts rather than installed console examples
- the package layout is partly namespace-style and partly traditional package-style

Those are maintainability issues, not correctness blockers, but they should be addressed in a follow-up cleanup pass.
