# Para-Core Usage Documentation

Para-Core is a flexible "agentic core" for running LLM agents with support for workflows, multi-stage execution, memory (short-term & long-term), and custom tools.

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
   - [Agents](#agents)
   - [Tools](#tools)
   - [Stages](#stages)
   - [Workflows](#workflows)
   - [Memory](#memory)
4. [Supported Models](#supported-models)

## Installation

Ensure you have Python 3.8+ installed.

```bash
pip install -r requirements.txt
```

*(Note: You may need to create a `requirements.txt` if one isn't provided, ensuring `httpx` and other dependencies are installed.)*

## Quick Start

Here is a minimal example of how to create and run a simple agent.

```python
from IkaCore.agents import IkaBaseAgent

# 1. Define the agent
agent = IkaBaseAgent(
    name="SimpleAssistant",
    description="A helpful assistant.",
    prompt="You are a helpful assistant. Answer the user's questions concisely.",
    model_id="gpt-4o",  # or claude-3-opus, deepseek-v3.2, etc.
    api_key="your-api-key-here",
    maxsteps=5
)

# 2. Run the agent
# The execution method starts the agent loop.
result = agent.execution()

print("Final Output:", result["final_message"])
print("Summary:", result["summary"])
```

## Core Concepts

### Agents

The `IkaBaseAgent` is the central class. It manages the interaction with the LLM, tool execution, and memory.

**Key Parameters:**
- `name` (str): Unique name for the agent.
- `description` (str): Description of the agent's purpose.
- `prompt` (str): The main system/task prompt.
- `model_id` (str): The ID of the model to use (e.g., "gpt-4o", "claude-3-sonnet").
- `api_key` (str): Your API key for the chosen provider.
- `tools` (List[IkaTools]): A list of tools available to the agent.
- `Stages` (List[IkaStage]): Optional list of stages for structured execution.
- `memory` (bool): Enable/disable memory features.
- `checkpoint` (bool): Enable/disable state checkpointing.

### Tools

Tools allow agents to perform actions like calling APIs or calculating values.

```python
from IkaCore.tools import IkaTools

def calculate_sum(args):
    try:
        a = float(args.get("a", 0))
        b = float(args.get("b", 0))
        return str(a + b)
    except Exception as e:
        return f"Error: {str(e)}"

# Define the tool
calc_tool = IkaTools(
    id="calculator",
    name="calculator",
    description="Calculates the sum of two numbers 'a' and 'b'.",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "First number"},
            "b": {"type": "number", "description": "Second number"}
        },
        "required": ["a", "b"]
    },
    execute_function=calculate_sum
)

# Add to agent
agent = IkaBaseAgent(
    ...,
    tools=[calc_tool]
)
```

### Stages

Stages allow you to break down a complex task into sequential steps, each with its own prompt and specific tools.

```python
from IkaCore.stages import IkaStage

# Stage 1: Research
research_stage = IkaStage(
    name="research",
    prompt="Research the topic provided by the user.",
    tools=[search_tool],  # specific tools for this stage
    stage_max_step=3
)

# Stage 2: Write
write_stage = IkaStage(
    name="write",
    prompt="Write a summary based on the research.",
    tools=[], 
    stage_max_step=2
)

# Agent with stages
agent = IkaBaseAgent(
    ...,
    Stages=[research_stage, write_stage]
)
```

### Workflows

Workflows allow you to connect multiple agents into a directed graph.

```python
from IkaCore.workflow import IkaWorkflow, WorkflowNode, WorkflowEdge

# Define nodes
node_a = WorkflowNode(name="researcher", agent=research_agent)
node_b = WorkflowNode(name="writer", agent=writer_agent)

# Define edges (flow)
edges = [
    WorkflowEdge(source="researcher", target="writer", edge_type="next")
]

# Create workflow
workflow = IkaWorkflow(
    name="ResearchPaperWorkflow",
    description="Researches and writes a paper.",
    nodes=[node_a, node_b],
    edges=edges
)

# Run workflow
results = workflow.run(initial_context="Topic: Quantum Computing")
print(results["writer"].final)
```

### Memory

Para-Core supports both Short-Term (session-based) and Long-Term (persistent) memory.

**Usage:**
1.  Initialize Global Memory (for Long-Term):
    ```python
    from IkaModel.base import init_global_long_term_memory
    
    init_global_long_term_memory(embedder_config={...})
    ```
2.  Enable in Agent:
    ```python
    agent = IkaBaseAgent(
        ...,
        memory=True,
        memory_access={
            "short_term_save": True,
            "short_term_search": True,
            "long_term_save": True,  # enable persistence
            "long_term_search": True
        }
    )
    ```

The agent will automatically have access to tools like `short_term_save`, `short_term_search`, etc.

## Supported Models

The framework supports multiple providers via `IkaModel`:

-   **OpenAI**: `gpt-4o`, `gpt-4.1`, `gpt-3.5-turbo`, etc.
-   **Anthropic**: `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`.
-   **Google**: `gemini-1.5-pro`, `gemini-2.0-flash`.
-   **DeepSeek**: `deepseek-v3.2`.

The system automatically detects the provider based on the `model_id`.
