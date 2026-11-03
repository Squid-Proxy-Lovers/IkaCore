# IkaMem Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         IkaMem                              │
│                     Memory System                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                            │
        ▼                                            ▼
┌──────────────────┐                        ┌──────────────────┐
│ ShortTermMemory  │                        │ LongTermMemory   │
├──────────────────┤                        ├──────────────────┤
│ - agent          │                        │ - agent          │
│ - task           │                        │ - task           │
│ - save()         │                        │ - save()         │
│ - search()       │                        │ - search()       │
│ - reset()        │                        │ - reset()        │
└────────┬─────────┘                        └─────────┬────────┘
         │                                            │
         │         ┌──────────────────┐              │
         └────────▶│  Memory (base)   │◀─────────────┘
                   ├──────────────────┤
                   │ - storage        │
                   │ - _agent         │
                   │ - _task          │
                   │ - save()         │
                   │ - search()       │
                   │ - reset()        │
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │ Storage (type)   │
                   ├──────────────────┤
                   │ + save()         │
                   │ + search()       │
                   │ + reset()        │
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │  Mem0Storage     │
                   ├──────────────────┤
                   │ - memory (Mem0)  │
                   │ - memory_type    │
                   │ - config         │
                   │ + save()         │
                   │ + search()       │
                   │ + reset()        │
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │   Mem0 Library   │
                   │  (mem0ai pkg)    │
                   └──────────────────┘
```

### Saving to Memory

```
User Code
   │
   │ memory.save(value, metadata)
   │
   ▼
ShortTermMemory.save()
   │
   │ creates ShortTermMemoryItem
   │ enriches data for Mem0
   │
   ▼
Memory.save() (base class)
   │
   │ delegates to storage
   │
   ▼
Mem0Storage.save()
   │
   │ formats as conversation
   │ builds parameters
   │ applies filters
   │
   ▼
Mem0 API / Local Mem0
   │
   │ semantic embedding
   │ storage
   │
   ▼
Memory stored
```

### Searching Memory

```
User Code
   │
   │ results = memory.search(query, limit, threshold)
   │
   ▼
ShortTermMemory.search()
   │
   │ logs search start
   │
   ▼
Memory.search() (base class)
   │
   │ delegates to storage
   │
   ▼
Mem0Storage.search()
   │
   │ builds search params
   │ applies filters (user_id, agent_id, run_id)
   │ sets metadata type filter
   │
   ▼
Mem0 API / Local Mem0
   │
   │ semantic similarity search
   │ metadata filtering
   │ graph-based reasoning
   │
   ▼
Results normalized
   │
   ▼
User Code receives results
```

## Class Hierarchy

```
Storage (interface)
    └── Mem0Storage

Memory (base)
    ├── ShortTermMemory
    └── LongTermMemory

(Plain classes - no inheritance)
    ├── ShortTermMemoryItem
    └── LongTermMemoryItem
```
