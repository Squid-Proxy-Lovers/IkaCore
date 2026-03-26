
This should server as both a guide to the IkaCore Agentic system and a gudie to agentic design and systems in general. If you would only like to see what is part IkaCore's API then that will be the start of each section. 


# IkaBaseAgent

```python
    def __init__(
        self,
        name: str,
        description: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        start_prompt: Optional[str] = None,
        end_prompt: Optional[str] = None,
        tools: Optional[List[IkaTools]] = None,
        model_id: str = "",
        api_key: str = "",
        api_url: Optional[str] = None,
        max_tokens: int = 20000,
        temperature: float = 0.0,
        checkpoint: bool = False,
        Batch: bool = False,
        BatchMax: int = 3,
        Stages: Optional[List[IkaStage]] = None,
        subagents: Optional[List["IkaBaseAgent"]] = None,
        next_agent: Optional["IkaBaseAgent"] = None,
        feedback_agent: Optional["IkaBaseAgent"] = None,
        maxsteps: int = 10,
        step_timeout: int = 900,
        rate_limit_per_min: Optional[float] = None,
        per_tool_rate_limit: Optional[Dict[str, float]] = None,
        RAGSource: Optional[List[type[IkaRAGSource]]] = None,
        memory: bool = False,
        memory_finder: Optional["Memory"] = None,
        memory_access: Optional[Dict[str, bool]] = None,
        final_answer_check: Optional[List[Callable]] = None,
        logging_level: int = 0,
        logging_file: str = "logs.txt",
        show_usage_level0: bool = True,
        checkpoint_db_path: str = "checkpoints.db",
        enable_summarization: bool = True,
        use_async: bool = False,
    ):
```

The foundation for the IkaCore system is this base agent API, while seeminly daunting at first glace the system is only as complex as you would like it to be. If you would like to keep the agent desgin and usage simple thats perfecly fine. 

```python
        self._validate_required_fields(
            {
                "name": name,
                "description": description,
                "prompt": prompt,
                "model_id": model_id,
                "api_key": api_key,
            }
        )
``` 
The only fields that are required is the above, quite simple right? 

The rest of the api is what makes this system so powerful and why we designed into such a way though. The primary point of IkaCore is to enable complex design of agents systems allowing for many interesting design choices and structures. This guide while hopefully server as a way of walking you through the entire code base as a user and a developer. 


## IkaTools

```python 
class IkaTools:
    def __init__(
        self, 
        name:str, 
        description:str, 
        parameters:dict, 
        limit_calls:int = 1, 
        required:bool = True, 
        execute_function:Callable = None, 
        parallel:bool = True, 
        id:str = None
    ) -> None:
```


If you haven't created agentic systems before this might be a new idea to you but there basic idea of tool calls in reference to agentic systems is that they server as a means of LLM's to actually use tools. Well what actually is a tool? Its anything that you may need to some level of external processing for that the LLM should have some level of control either when to execute and/or arguments to provide it based on previous execution context. Lets say you have an agent whose sole purpose is to figure out what files are in our current directory and what they do. In order to do this you like would want some simple tools such as read and list. To do that all we need to do is the following (example_fileagent.py): 

```python
    get_pwd = IkaTools(
        name="get_pwd",
        description="Get the current working directory",
        parameters={
            "None": "No parameters are required",
        },
        execute_function=lambda _: os.getcwd(),
    )
    
    list_files = IkaTools(
        name="list_files",
        description="List files in a directory",
        parameters={
            "directory_path": "The path to the directory to list files from",
        },
        execute_function=lambda x: os.listdir(x["directory_path"]),
    )

    read_file = IkaTools(
        name="read_files",
        description="Read file content",
        parameters={
            "file_path": "The path to the file to read",
        },
        execute_function=lambda x: os.open(x["file_path"], "r").read(),
    )

```

In terms of required fields the following is needed: 

```python 
        validate_required_fields(
            {
                "name": name,
                "description": description,
                "parameters": parameters,
                "execute_function": execute_function,
            }
        )
```

Now in order to pass that into our agent all we need to do is the following: 
```python
    agent = IkaBaseAgent(
        name="example_system",
        description="A simple example system",
        system_prompt=SYSTEM_PROMPT,
        prompt=PROMPT,
        tools=[get_pwd, read_file, list_files],
        model_id=MODEL_ID,
        api_key=API_KEY,
    )
```

What happens on the backend when we create these tool calls is a structured JSON is passed into our LLM provider API detailing the to see this we are going to want to set the agent logging level to `logging_level=3`. Giving us the following output: 

```json
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_pwd",
        "description": "Get the current working directory",
        "parameters": {
          "type": "object",
          "properties": {
            "None": {
              "type": "string",
              "description": "No parameters are required"
            }
          },
          "required": []
        }
      }
    },
...
    {
      "type": "function",
      "function": {
        "name": "agent_end",
        "description": "End the agent loop with a final answer. Provide the final output and any key reasoning. CRITICAL: You MUST provide your final answer in the 'input' parameter. Do NOT call this tool with empty arguments.",
        "parameters": {
          "type": "object",
          "properties": {
            "input": {
              "type": "string",
              "description": "Final response content. This is REQUIRED - provide your complete final answer here."
            }
          },
          "required": [
            "input"
          ]
        }
      }
    }
  ],
```

Did the `required` array catch your eye? Well good, this allows you to enforce which arguments are required to be generated when using a tool call in order to use this for your own tools you have to pass it in the params dictionary, by default we assume its optional and a string type: 
```python
    list_files = IkaTools(
        name="list_files",
        description="List files in a directory",
        parameters={
            "directory_path": {
                "type": "string",
                "description": "The path to the directory to list files from",
                "required": True
            },
        },
        execute_function=lambda x: os.listdir(x["directory_path"]),
    )
```

In regards to the other args: 

**limit_calls (int, default=0):** 
- Adds an execution limit for tool calls per agent or per stage based on if you have stages or not. the default is 0 which internally means no limit

**required (bool, default=True):** 
- The API will enforce that the agent calls this tool (or at least one tool if multiple are required), rather than allowing it to generate a regular message response.

**parallel (bool, default=True):**
- Controls whether this tool can execute concurrently with other tools; when False, the tool executes sequentially, which is useful for tools that must run in order or have dependencies.

**id (str, optional):** 
- A unique identifier for the tool; if not provided, a UUID is auto-generated to track and reference the tool internally.



----
## IkaStage
```python
    def __init__(
        self,
        name: str,
        prompt: str,
        tools: List[IkaTools],
        stage_max_step: int = 1,
        subagents: Optional[list] = None,
        allowed_back_to: Optional[List[int]] = None,
        hitl: bool = False,
        memory_access: Optional[Dict[str, bool]] = None,
        long_term_filter: Optional[Callable[..., Any]] = None,
        checkpoint: bool = False,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ):
```

The most novel aspect of the IkaCore agentic system is our creation of the "stage" execution system. While our team was creating complex agentic systems we found that we often saw ourselves creating very detailed and defined execution "stages" via the prompt. My idea was why leave this as purely a prompt based system - instead I define an actual execution system which allows us to define several things. The first thing this allows us to do is have far stricter tool call control and agent control, more often then not while we where testing systems and design choices our agent would start using the wrong tool during specific execution stages. Say we have an agent thats goal is to analyze a complex code base and then write these findings to some kinda memory layer to be used later. There are at least two clear stages, a read + analysis stage and a write stage - both of these stages will never need access to the others tool calls but we don't want them to be distinct "agents". I think whats key to remember in all of this the backend API's are all just stateless API's so all of these abstraction layers main goal is going to be allowing for more granular control of execution dependent on requirements. 

Something fun we can do is swap model providers during stage execution - one of the nice things about IkaCore is I fully removed all the backend SDK's and created a uniform execution layer allowing easy movement between API's since we define and store everything in our abstraction layers then only translate to the APIs at the very end. 

```python
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
```

If no values are provided for the model specific data then we will simply assume the default provided by the agent, inputing them as seen is purely optional. This allows for some fun things like having specific sections in your execution stages that are very high effort or low effort.

```python
        name: str,
        prompt: str,
        tools: List[IkaTools],
        stage_max_step: int = 1,
        subagents: Optional[list] = None,
```

The arguments `tools` and `name` are very stright forward, tools are what tools your stage actually gets to see while the name is what your going to see on your cli output as well as what gets appended to the prompt. The `stage_max_step` is a counter for the stage, when you hit the counter you will move to the next stage. While there are per stage counters these stage counters are also still part of the global agent counter. If your agent counter is say 100 but you expect 5 stages of 25 then your system won't work. 

Now the `prompt` section is some what straight forward but its key to understand how prompts for Stages work in IkaCore.  We are going to dynamically general the final prompt used by the system, when a stage it will be added in linear order as you defined and the current stage will be told to the system. This way it understands what it has to do know as well as what it will do afterwards. All you have to worry about as a user is including the prompt. Its key to understand this is why we have given the base agent the `start_prompt` and `end_prompt`. We will create the middle section using the parts provided its the user job only to make the start and end. 


```
┌──────────────────────────────────────────────────────────────────────────────┐
│ [AGENT INIT] Step 0 | Thread:8455151808 Instance:0                           │
├──────────────────────────────────────────────────────────────────────────────┤
│ Chain: example_system -> Stage 0: list_stage                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│ Content Prompt:                                                              │
│                                                                              │
│ You are a simple example system. You can read and list files,                │
│ your goal is to return a summary of the files in the current working         │
│ directory,                                                                   │
│ You must excute all the stages as defined by the system prompt.              │
│                                                                              │
│                                                                              │
│ STAGES:                                                                      │
│                                                                              │
│ Stage 0 (list_stage):                                                        │
│                                                                              │
│ In this stage you will need to list the files in the current working         │
│ directory.                                                                   │
│ You will need to use the list_files tool to achieve your goal.               │
│ you should explore the files in the current working directory and            │
│ subdirectories.                                                              │
│                                                                              │
│ after you have listed the files, you should move to the next stage.          │
│                                                                              │
│ you should only use get_pwd tool once in this stage.                         │
│                                                                              │
│                                                                              │
│ Stage 1 (read_stage):                                                        │
│                                                                              │
│ In this stage you will need to read the files in the current working         │
│ directory.                                                                   │
│ You will need to use the read_file tool to achieve your goal.                │
│ You will need to return the content of the files that you have listed in the │
│ previous stage.                                                              │
│                                                                              │
│ after you have read the files, you should move to the next stage.            │
│                                                                              │
│ you should only use get_pwd tool once in this stage.                         │
│                                                                              │
│                                                                              │
│ Stage 2 (summarize_stage):                                                   │
│                                                                              │
│ In this stage you will need to summarize the content of the files in the     │
│ current working directory.                                                   │
│ You will need to return the summary of the files in the current working      │
│ directory.                                                                   │
│ If you have read the files in the previous stage, you should use the content │
│ of the files to summarize the files.                                         │
│                                                                              │
│                                                                              │
│ CURRENT STAGE IS:                                                            │
│                                                                              │
│                                                                              │
│ In this stage you will need to list the files in the current working         │
│ directory.                                                                   │
│ You will need to use the list_files tool to achieve your goal.               │
│ you should explore the files in the current working directory and            │
│ subdirectories.                                                              │
│                                                                              │
│ after you have listed the files, you should move to the next stage.          │
│                                                                              │
│ you should only use get_pwd tool once in this stage.                         │
│                                                                              │
│                                                                              │
│                                                                              │
│ STAGE MOVEMENT:                                                              │
│ Use the stage_end tool when the current stage is complete to advance to the  │
│ next stage.                                                                  │
│ If the change_stage tool is available, use it with stage_index (the stage to │
│ change to) and reason (why you need to go back). stage_index must be in that │
│ stage's allowed_back_to list.                                                │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```
 *you can see this in the example provided in example_stages* 



Now on to some of the weirder aspects of stages that allow us to do some funny things  
```python
        allowed_back_to: Optional[List[int]] = None,
        hitl: bool = False,
        memory_access: Optional[Dict[str, bool]] = None,
        long_term_filter: Optional[Callable[..., Any]] = None,
        checkpoint: bool = False,
```



The most important tool here is the HITL mode, it allows you to define a libarary supported input phase allowing for human in the loop behavior. HITL exists as primarily as a tool call which the agent uses ask_user only when it needs an answer. The agent sends a question in the tool call; the user is prompted in orange and the reply is returned as the tool result. We also will append a section to the stage prompt on how to use HITL so the agent has good context. I still strongly recommend including in your own stage prompt that we need this. 

```
[HITL] stage=read_stage question='Which specific files would you like me to read from the directory? The available files are: .cursor, .DS_Store, Example, .claude, docs, .mypy_cache, examples, .git, src.'
[HITL] Which specific files would you like me to read from the directory? The available files are: .cursor, .DS_Store, Example, docs, .mypy_cache, examples, .git, src.
Your answer: 
```

`allowed_back_to` is a pivotal argument to understand how stages work. It essentially functions as a control flow mechanism. Stages are allowed to move backwards to any stage listed in the allowed_back_to section. This exists as a tool call but the idea is maybe one stage is a verification stage and its not happy with the amount of data it has so it wants to go back to the read stage. Internally our code base will move stages linearly allowing the agent to decides when to move stages, this means forward execution is not controllable/skippable but backwards execution is. 

```
        memory_access: Optional[Dict[str, bool]] = None,
        long_term_filter: Optional[Callable[..., Any]] = None,
```

The two arguments above are directly connected to each other as they exists as systems for accessing the memory layer of the agents. We provide integration to mem0 allowing for saving long term data and searching through it. Instead of making the user have to integrate tools for this we just provide them and enable them with the following arguments. The `long_term_filter`  applies filters for specific long term information and we use the `memory_access` to set the following values:
```
short_term_save	    Enables short_term_save (write to short‑term memory).
short_term_search	Enables short_term_search (search short‑term memory).
long_term_save	    Enables long_term_save (write to long‑term memory).
long_term_search	Enables long_term_search (search long‑term memory).
```

Finally we go to `checkpoint`, which are quite fun they allow for execution based snapshot of your agentic work flow and to restart execution form where you took a snapshot by passing in a generate UID. The UID will be created and outputted during execution flow and stored in a sqlitedb. Its very important to note that you must set the AGENTS checkpoint flag as well as the stage checkpoint flag or checkpoints will not work at a stage level. In order to resume execution states all we need to do is pass it in as a arg for the execution function: 
- `result = agent.execution(checkpoint_uid="f47ac10b-58cc-4372-a567-0e02b2c3d479")`
```
┌──────────────────────────────────────────────────────┐
│ [AGENT INIT] Step 0 | Thread:8455151808 Instance:0   │
├──────────────────────────────────────────────────────┤
│ Chain: example_system -> Stage 1: read_stage         │
├──────────────────────────────────────────────────────┤
│ Checkpoint saved at Stage 1: read_stage              │
│ Checkpoint UID: 41272d37-d976-49a1-a0c6-6c289ef9b9f8 │
│ Remaining steps: 76                                  │
└──────────────────────────────────────────────────────┘
```
 *example of a snapshot* 



## IkaWorkflow

The second execution framework we allow in IkaCore for agents is "Workflows" which are directed acyclic graphs based workflow where each node is an IkaBaseAgent. It supports sync and async execution, next (sequential) and child (sub‑tasks) edges, context compression between nodes, and parallel instances of the same node. The idea here is many times you don't want an agentic system to control its own execution so we allow you to define your own execution and enforce this system on agents. 

```python
class IkaWorkflow:
    def __init__(
        self,
        name: str,
        description: str,
        nodes: List[WorkflowNode],
        edges: List[WorkflowEdge],
        compress_hook: Optional[WorkflowCompressionHook] = None,
        start_node: Optional[str] = None,
        async_executor: Optional[AsyncWorkflowExecutor] = None,
        max_parallel_workers: int = 30,
    ):
```

The IkaCore workflow API starts with three data structures: WorkflowEdge, WorkflowNode, and WorkflowResult. The workflow itself is built from nodes and edges; run returns a dict of node name to WorkflowResult.

### WorkflowEdge

```python
@dataclass
class WorkflowEdge:
    source: str
    target: str
    edge_type: str = "next"
    stage_index: Optional[int] = None
```

Edges define execution flow between nodes. `source` and `target` are node names. `edge_type` must be `"next"` or `"child"`:

**`edge_type="next"`**  
Forward flow. The target runs after the source. The target receives the source's output (and any child outputs) as compressed context via the `compress_hook`. Next edges drive the dependency graph used for async execution and ordering.

**`edge_type="child"`**  
Sub-task of the source. The target runs during the source: the workflow runs it after the source's agent finishes, passes the source's summary as context, and folds the child's summary into the source's downstream context for any next edges. Control returns to the parent; the child does not have its own next successors in the main DAG unless you also add a next edge from it.

**`stage_index`**  
Only meaningful for `child` edges. When the source is a staged agent (has `Stages`), `stage_index` binds the child's agent to that stage. The workflow's `_bind_stage_wiring` adds the child to that stage's `subagents`, so the stage can call it as a tool. The child is also run as a normal workflow node after the parent; `stage_index` only affects wiring.

Linear example:

```python
edges = [
    WorkflowEdge(source="node_a", target="node_b", edge_type="next"),
]
```

Staged parent with children and a next:

```python
edges = [
    WorkflowEdge(source="orchestrator", target="researcher", edge_type="child", stage_index=0),
    WorkflowEdge(source="orchestrator", target="critic", edge_type="child", stage_index=1),
    WorkflowEdge(source="orchestrator", target="synthesizer", edge_type="next"),
]
```

### WorkflowNode

```python
@dataclass
class WorkflowNode:
    name: str
    agent: IkaBaseAgent
    stage_wiring: Optional[Dict[int, Dict[str, List[IkaBaseAgent]]]] = None
    instances: int = 1
    instance_inputs: Optional[List[str]] = None
```

`name` identifies the node in edges and `start_node`. `agent` is the IkaBaseAgent to run. `stage_wiring` is usually filled by the workflow from child edges with `stage_index`; you rarely set it by hand.

**`instances` (default 1)**  
In async mode (`run(..., use_async=True)`), how many copies of this node run in parallel. Each copy is a deepcopy of the agent.

**`instance_inputs`**  
Optional list of strings, one per instance. When set, instance `i` gets `instance_inputs[i]` as its prompt override (and still receives workflow context). Useful to run the same agent in parallel with different tasks.

### WorkflowResult

```python
@dataclass
class WorkflowResult:
    name: str
    final: str
    summary: str
    history: Dict
    child_summaries: Dict[str, str] = field(default_factory=dict)
```

`run` and `run_async` return `Dict[str, WorkflowResult]`.

- **`final`**: `execution_output["final_message"]` from the agent.
- **`summary`**: `execution_output["summary"]` or `final`; this is what gets passed downstream and into the compress hook. With `summarize_final=True` on the agent, summary is a compressed version of the conversation; otherwise it falls back to `final_message`.
- **`history`**: the raw `agent.execution()` return, i.e. `{"final_message": ..., "summary": ...}`. It does not include the full message history.
- **`child_summaries`**: for nodes with child edges, `{child_node_name: child_result.summary}`. Used when building the parent's downstream context for next edges: the compress hook receives `[summary] + list(child_summaries.values())`.

### compress_hook and run

**`compress_hook`**  
`WorkflowCompressionHook = Callable[[List[str], IkaBaseAgent], str]`. It takes a list of upstream (and child) summary strings and the target agent, and returns one string used as that node's workflow context. The default `_default_compress_hook` merges the strings and runs `summarise_message_history` to shorten them; on failure it returns the merged text. You can pass a custom hook (e.g. truncation or a different summarizer).

**`run(initial_context=None, use_async=False)`**  
- **Sync** (`use_async=False`): DFS from `start_node`. For each node: apply stage wiring, inject context, run the agent, run all child nodes and collect their summaries, then follow next edges with the compressed `[summary] + child_summaries`. Child edges and `stage_index` are only used in sync.
- **Async** (`use_async=True`): Uses only `next` edges for the dependency graph. Runs nodes when their upstream deps are done; supports `instances` and `instance_inputs` for parallel copies. Does not run child edges.

**`initial_context`**  
Injected as context for the start node. For sync it is added to `upstream_contexts[start_node]`; for async the start node gets it when it has no upstream yet.

### Minimal example

From `examples/example_workflow.py`:

```python
agent_a = IkaBaseAgent(
    name="agent_a",
    description="First agent in the pipeline",
    system_prompt="You are a concise assistant. Reply in one short paragraph.",
    prompt="Expand on the topic you receive. Produce a brief explanation.",
    tools=[],
    model_id=MODEL_ID,
    api_key=API_KEY,
)

agent_b = IkaBaseAgent(
    name="agent_b",
    description="Second agent in the pipeline",
    system_prompt="You are a concise assistant. Reply in one short paragraph.",
    prompt="Summarize the text you receive into 2 or 3 bullet points.",
    tools=[],
    model_id=MODEL_ID,
    api_key=API_KEY,
)

node_a = WorkflowNode(name="node_a", agent=agent_a)
node_b = WorkflowNode(name="node_b", agent=agent_b)

edges = [
    WorkflowEdge(source="node_a", target="node_b", edge_type="next"),
]

workflow = IkaWorkflow(
    name="two_step",
    description="Linear workflow: A then B",
    nodes=[node_a, node_b],
    edges=edges,
    start_node="node_a",
)

results = workflow.run(initial_context="The number 42 and its cultural significance.")

for name, res in results.items():
    print(f"[{name}] summary: {res.summary[:200]}...")
print("Final from last node:", results["node_b"].final[:300], "...")
```

For staged parents, child edges with `stage_index`, a custom `compress_hook`, and async with `instances` and `instance_inputs`, see `examples/example_workflow_advanced.py`.
