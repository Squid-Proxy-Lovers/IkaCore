# IkaMem
memory system using mem0 (https://mem0.ai)

## install
```bash
pip install mem0ai
export MEM0_API_KEY='your_key'
```

## short-term memory
```python
from IkaMem import STMemory

config = {
    "provider": "mem0",
    "config": {
        "api_key": "<mem0_api_key>",
        "user_id": "user_123",
        "agent_id": "agent_456"
    }
}

memory = STMemory(embedder_config=config)
memory.agent = "MyAgent"

memory.save("user prefers detailed explanations", metadata={"type": "preference"})

results = memory.search("what does user prefer?", limit=5)
for r in results:
    print(r['content'])
```

## long-term memory
```python
from IkaMem import LTMemory, LTMemItem
from datetime import datetime

config = {
    "provider": "mem0",
    "config": {
        "api_key": "<mem0_api_key>",
        "user_id": "user_123"
    }
}

memory = LTMemory(embedder_config=config)

item = LTMemItem(
    agent="MyAgent",
    task="analyze market trends",
    expected_output="market analysis report",
    datetime=datetime.now().isoformat(),
    quality=0.95
)
memory.save(item)

results = memory.search("previous analysis tasks", limit=3)
```

## basic usage
```python
from IkaMem import STMemory

config = {
    "provider": "mem0",
    "config": {"api_key": "<your_key>", "user_id": "user_123"}
}

mem = STMemory(embedder_config=config)
mem.agent = "Assistant"
mem.task = "help with coding"

mem.save("user working on FastAPI project")
mem.save("user prefers type hints")
mem.save("project follows MVC pattern")

context = mem.search("user's project", limit=5)
```

## local mem0 (no api key)
```python
config = {
    "provider": "mem0",
    "config": {
        "local_mem0_config": {
            # local mem0 config
        }
    }
}

memory = STMemory(embedder_config=config)
```

## config options
```python
config = {
    "provider": "mem0",
    "config": {
        "api_key": "...",           # or MEM0_API_KEY env var
        "user_id": "...",           # user identifier
        "agent_id": "...",          # agent identifier
        "run_id": "...",            # run identifier (short-term)
        "org_id": "...",            # organization id
        "project_id": "...",        # project id
        "local_mem0_config": {},    # local config
        "infer": True,              # auto-infer categories
        "includes": [],             # categories to include
        "excludes": [],             # categories to exclude
        "custom_categories": []     # custom categories
    }
}
```

