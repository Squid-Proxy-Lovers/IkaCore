# IkaMem Architecture

`STMemory` and `LTMemory` share the `Memory` base class. Both require an explicit
storage backend, or `embedder_config={"provider": "mem0", "config": ...}` to create
a `Mem0Store`. Mem0 is optional; constructing `Mem0Store` without it raises
`ImportError`. Custom backends work without Mem0.

## Current types

- `Memory` owns a `StorageBackend` and the optional `agent` and `task` labels.
- `STMemory` wraps saved values in `STMemItem`, then passes its data and metadata
  to storage. With Mem0 it adds a short instruction to the stored text.
- `LTMemory` accepts ordinary values or `LTMemItem`. For an item it stores the task
  as the value and enriches metadata with agent, expected output, quality and datetime.
  Search applies the default five-result filter after retrieving candidates.
- `STMemItem` and `LTMemItem` describe data and metadata; applications can supply
  their own metadata without a dynamically generated memory schema.
- `StorageBackend` in `memory.py` is the structural protocol for synchronous
  `save(value, metadata)`, `search(query, limit, score_threshold)` and `reset()`.
- `Storage` in `storage/interface.py` is an abstract interface with the same methods.
- `Mem0Store` implements `Storage` using a hosted or local Mem0 client. It normalizes
  search results, propagates identity/configuration filters and labels memory type.

## Data flow

Saving flows from `STMemory.save()` or `LTMemory.save()` through the backend's
`save()` method. Searching flows through the backend's `search()` and then the
memory wrapper's result handling/filter. `reset()` delegates to the backend.
`Memory` itself also supports direct save/search without the item-specific wrapping.

`IkaMem.agent_tools.MemoryTools` adapts explicit backends to ordinary `IkaTools`.
Pass `.tools(access=..., long_term_filter=...)` to an agent or stage's `tools`.
Core agents have no memory-specific schema, executor or backend lookup. Construct
`STMemory(storage=...)` / `LTMemory(storage=...)`, or use `embedder_config` for Mem0,
and share the same backend explicitly when cross-run memory is desired.
Long-term save uses structured task/output/metadata and creates an `LTMemItem`.
Search tools allow replay; saves deny replay unless the caller approves repeating effects.

## Backend boundaries

The public storage protocol is synchronous; native async storage methods are not
part of this interface. Custom backends must implement all three protocol methods.
Offline backend and agent integration examples are in the maintained
[usage guide](../../../docs/USAGE.md#memory); hosted Mem0 setup is in the
[memory guide](README.md). Hosted Mem0 checks require separately configured credentials.
