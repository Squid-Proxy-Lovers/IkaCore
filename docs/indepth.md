## IkaCore Architecture Notes

This document explains how the current IkaCore runtime is structured and where the main execution boundaries live.

### Package Layout

- `src/IkaCore`: agent runtime, stages, workflow graph, logging, snapshots
- `src/IkaModel`: provider payload builders, transport layer, tool execution loop, summarization helpers
- `src/IkaMem`: short-term and long-term memory abstractions
- `src/IkaTest`: regression tests for runtime behavior

### Core Runtime Model

`IkaBaseAgent` is the public entry point. It mixes together four responsibilities:

- `AgentMemoryMixin`: memory initialization and memory tool helpers
- `AgentToolsMixin`: tool conversion, stage tool assembly, subagent delegation
- `AgentExecutionMixin`: thin execution entry points
- `AgentHelpersMixin`: shared runtime helpers, validation, snapshot and HITL support

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

Codex auth is intentionally not automatic in the request path. The Codex provider treats `BareBoneModel.api_key` as the literal bearer token. Users can pass their own bearer token, or explicitly import `get_bearer` with `from IkaModel.codex.auth import get_bearer` and call `get_bearer()` to read and refresh the Codex CLI credentials from `~/.codex/auth.json` or `$CODEX_HOME/auth.json`.

### HITL And Resume

Human-in-the-loop no longer depends on blocking terminal input.

Current behavior:

- `ask_user` raises a structured interrupt
- the interrupt can be saved as a snapshot
- `resume_from_snapshot(snapshot_id, resume_input=...)` continues the run

Regression coverage lives in [`test_hitl_interrupt_resume.py`](../src/IkaTest/test_hitl_interrupt_resume.py).

### Snapshot Persistence and Replay

[`SnapshotStore`](../src/IkaCore/snapshot_store.py) is the sole persistence backend.
Call `agent.enable_snapshots(path, control)` or `workflow.enable_snapshots(path, control)`
to opt in. `RuntimeControl` selects pause points; execution returns a `snapshot_id`
when paused or waiting for human input. An equivalently configured agent can open the
same database and call `resume_from_snapshot(snapshot_id, resume_input=...)`.

The SQLite model has three related records:

- A run identifies an agent or workflow execution and its status.
- Frames describe nested agent, stage, tool, subagent and workflow-instance execution,
  including lineage, inputs, outputs and resolved configuration.
- Immutable snapshots capture frame state, transcript and usage at boundaries. Events
  record lifecycle and tool activity for inspection and replay assessment.

A boundary's snapshot, event and requested pause status commit atomically. Connections
are scoped to operations/transactions; no database lock spans a provider or tool call.
Post-model resume consumes the saved response without counting usage twice, and completed
snapshots return their saved output. `SnapshotStore.fork_run_from_snapshot()` creates a
new run with cloned lineage; `restart_frame()` forks from the frame's entry snapshot.
Replay checks tool-effect safety and requires `allow_unsafe_replay=True` for unsafe effects.
It cannot restore a Python stack or undo external side effects.

Workflow `resume_from_snapshot()` reuses completed instance results. Selective instance
replay invalidates downstream results; nested or unfinished tool effects are checked
before replay. `snapshot_runtime_info()` exposes run/frame IDs and the last snapshot ID.
Snapshot JSON filters known credential patterns; redacted state is refused for replay
by default. Keep databases private because filtering cannot identify all private content.

V2 removes `CheckpointStore`, `load_checkpoint()`, `checkpoint_uid` and constructor/stage
checkpoint options. Old checkpoint databases are left intact and are not converted;
finish those runs with v1 before migrating. See the [migration guide](v2/README.md#breaking-changes-and-migration).

### Memory

`IkaMem` exposes:

- `STMemory`
- `LTMemory`
- `Mem0Store`

Agent-side memory tools are assembled by `AgentMemoryMixin`. Long-term save and search now use structured tool schemas instead of the old string-splitting format.

### Maintenance Gates

The repo now has explicit gates for the maintenance risks that used to drift:

- production functions are limited to 70 lines and modules to 600 lines; there is no whole-class size limit (enforced by `test_quality_gates.py` in `scripts/quality.py`)
- broad `except Exception` boundaries are limited to the current intentional runtime isolation points
- `pyright` checks import/name, call, argument, assignment, return, optional, iterable, and attribute access regressions
- coverage must stay above the configured project threshold
- optional live provider smoke tests are kept separate from deterministic CI

The remaining maturity gap is not local unit behavior. It is broader integration confidence: real provider smoke coverage is opt-in, and long-running degraded-network workflows still need periodic manual or scheduled runs with credentials.
