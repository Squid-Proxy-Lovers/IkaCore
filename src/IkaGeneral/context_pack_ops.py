"""Mixin providing context-pack meta-tool executors."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import TYPE_CHECKING

from .helpers import estimate_tokens

if TYPE_CHECKING:
    from .execution_env import IkaExecutionEnvironment


class ContextPackMixin:
    """Executor methods for context-pack meta-tools.

    Mixed into :class:`IkaExecutionEnvironment`.
    """

    def _default_context_book(self: "IkaExecutionEnvironment") -> str:
        store = getattr(self, "_ccp_context_store", None)
        if store is not None:
            return store.book_name
        return "context_packs"

    def _context_pack_key(
        self: "IkaExecutionEnvironment",
        name: str,
        book: str | None = None,
    ) -> str:
        return f"{(book or self._default_context_book()).strip()}::{name.strip()}"

    def _register_context_pack(
        self: "IkaExecutionEnvironment",
        pack,
    ) -> str:
        key = self._context_pack_key(pack.name, getattr(pack, "book_name", None))
        self._context_pack_registry[key] = pack
        return key

    def _resolve_context_pack_key(
        self: "IkaExecutionEnvironment",
        name: str,
        book: str | None = None,
    ) -> str | None:
        if not name:
            return None
        if "::" in name:
            book_part, pack_part = name.split("::", 1)
            key = self._context_pack_key(pack_part, book_part)
            return key if key in self._context_pack_registry else None
        if "/" in name and book is None:
            book_part, pack_part = name.split("/", 1)
            key = self._context_pack_key(pack_part, book_part)
            return key if key in self._context_pack_registry else None

        key = self._context_pack_key(name, book)
        if key in self._context_pack_registry:
            return key

        matches = [
            registry_key
            for registry_key, pack in self._context_pack_registry.items()
            if pack.name == name
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    def _parse_context_pack_ref(
        self: "IkaExecutionEnvironment",
        args: dict,
    ) -> tuple[str, str]:
        name = args.get("name", "").strip()
        book = (args.get("book") or "").strip()
        if "::" in name:
            book, name = name.split("::", 1)
        elif "/" in name and not book:
            book, name = name.split("/", 1)
        return name, (book or self._default_context_book())

    def _drop_context_pack_from_runtime(
        self: "IkaExecutionEnvironment",
        name: str,
        book: str,
    ) -> bool:
        pack_key = self._resolve_context_pack_key(name, book) or self._context_pack_key(name, book)
        removed = self._context_pack_registry.pop(pack_key, None) is not None
        self._loaded_packs = [key for key in self._loaded_packs if key != pack_key]
        return removed

    # ── create ─────────────────────────────────────────────────────────────

    async def _exec_create_context_pack(
        self: "IkaExecutionEnvironment", args: dict
    ) -> str:
        from .models import ContextPack

        name = args.get("name", "")
        content = args.get("content", "")
        tags_str = args.get("tags", "")
        book = (args.get("book") or self._default_context_book()).strip()
        tags = (
            [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
        )
        key = self._context_pack_key(name, book)
        replaced_existing = False
        fingerprint_key = self._root_plan_pack_fingerprint_key(name, book)
        fingerprint = self._compute_plan_pack_fingerprint(content) if fingerprint_key else None

        if fingerprint_key and fingerprint is not None:
            previous = self._plan_pack_write_fingerprints.get(fingerprint_key)
            if previous == fingerprint:
                return json.dumps({
                    "status": "blocked_duplicate_create",
                    "error": (
                        "CRITICAL: STOP. Root already wrote this plan pack with materially identical "
                        "content during this execution. Do not call create_context_pack again for it."
                    ),
                    "name": name,
                    "book": book,
                    "next_required_action": (
                        "Load the existing pack if you need to inspect it. Only rewrite it if the plan has "
                        "materially changed; otherwise continue execution instead of rewriting the plan."
                    ),
                    "stop_repeating_tool_call": {
                        "tool": "create_context_pack",
                        "name": name,
                        "book": book,
                        "reason": "same_execution_same_content",
                    },
                })

        async with self._context_lock:
            if key in self._context_pack_registry:
                self._drop_context_pack_from_runtime(name, book)
                replaced_existing = True
                if self._ccp_context_store is not None:
                    await asyncio.to_thread(self._ccp_context_store.delete_pack, name, book)
            pack = ContextPack(
                name=name,
                content=content,
                book_name=book,
                tags=tags,
                source="agent",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                size_tokens=estimate_tokens(content),
            )
            if self._ccp_context_store is not None:
                pack = await asyncio.to_thread(self._ccp_context_store.upsert_pack, pack)
            self._register_context_pack(pack)
            if fingerprint_key and fingerprint is not None:
                self._plan_pack_write_fingerprints[fingerprint_key] = fingerprint

        return json.dumps({
            "status": "recreated" if replaced_existing else "created",
            "name": name,
            "book": book,
            "size_tokens": pack.size_tokens,
            "replaced_existing": replaced_existing,
        })

    # ── load ───────────────────────────────────────────────────────────────

    def _exec_load_context_pack(
        self: "IkaExecutionEnvironment", args: dict
    ) -> str:
        name, book = self._parse_context_pack_ref(args)
        chunk = args.get("chunk")

        pack_key = self._resolve_context_pack_key(name, book)
        pack = self._context_pack_registry.get(pack_key) if pack_key else None
        if self._ccp_context_store is not None:
            try:
                pack = self._ccp_context_store.get_pack(name, book)
                pack_key = self._register_context_pack(pack)
            except Exception as exc:
                if pack is None:
                    return json.dumps({"error": f"Context pack '{name}' not found in book '{book}' in CCP: {exc}"})
        elif pack is None:
            return json.dumps({"error": f"Context pack '{name}' not found in book '{book}'"})

        if pack_key in self._loaded_packs:
            return json.dumps({"status": "already_loaded", "name": name, "book": book})

        # Chunked loading for large packs
        content = pack.content
        has_more = False
        if chunk is not None:
            chunk_size = 50_000  # ~12.5 K tokens per chunk
            chunk_idx = int(chunk)
            start = chunk_idx * chunk_size
            end = start + chunk_size
            content = pack.content[start:end]
            has_more = end < len(pack.content)

        assert pack_key is not None
        self._loaded_packs.append(pack_key)

        # Inject into conversation
        if self._messages is not None:
            self._messages.append({
                "role": "user",
                "content": f"[Context Pack Loaded: {book}/{name}]\n{content}",
            })

        result: dict = {
            "status": "loaded",
            "name": name,
            "book": book,
            "size_tokens": estimate_tokens(content),
        }
        if has_more:
            result["has_more_chunks"] = True
            result["next_chunk"] = int(chunk) + 1  # type: ignore[arg-type]
        return json.dumps(result)

    # ── unload ─────────────────────────────────────────────────────────────

    def _exec_unload_context_pack(
        self: "IkaExecutionEnvironment", args: dict
    ) -> str:
        name, book = self._parse_context_pack_ref(args)
        pack_key = self._resolve_context_pack_key(name, book) or self._context_pack_key(name, book)
        if pack_key not in self._loaded_packs:
            return json.dumps({"error": f"Context pack '{name}' in book '{book}' is not loaded"})
        self._loaded_packs.remove(pack_key)
        return json.dumps({"status": "unloaded", "name": name, "book": book})

    # ── delete ─────────────────────────────────────────────────────────────

    async def _exec_delete_context_pack(
        self: "IkaExecutionEnvironment", args: dict
    ) -> str:
        name, book = self._parse_context_pack_ref(args)

        async with self._context_lock:
            removed_runtime = self._drop_context_pack_from_runtime(name, book)
            removed_remote = False
            if self._ccp_context_store is not None:
                removed_remote = await asyncio.to_thread(
                    self._ccp_context_store.delete_pack,
                    name,
                    book,
                )

        if not removed_runtime and not removed_remote:
            return json.dumps({
                "status": "not_found",
                "name": name,
                "book": book,
            })

        return json.dumps({
            "status": "deleted",
            "name": name,
            "book": book,
        })

    # ── update ─────────────────────────────────────────────────────────────

    async def _exec_update_context_pack(
        self: "IkaExecutionEnvironment", args: dict
    ) -> str:
        name, book = self._parse_context_pack_ref(args)

        async with self._context_lock:
            pack_key = self._resolve_context_pack_key(name, book)
            if pack_key is None:
                return json.dumps({"error": f"Context pack '{name}' not found in book '{book}'"})

            pack = self._context_pack_registry[pack_key]
            created_at = pack.created_at
            if args.get("content"):
                pack.content = args["content"]
                pack.size_tokens = estimate_tokens(args["content"])
            if args.get("tags"):
                pack.tags = [
                    t.strip() for t in args["tags"].split(",") if t.strip()
                ]
            pack.updated_at = datetime.now()
            self._drop_context_pack_from_runtime(name, book)
            if self._ccp_context_store is not None:
                await asyncio.to_thread(self._ccp_context_store.delete_pack, name, book)
            pack.created_at = created_at
            if self._ccp_context_store is not None:
                pack = await asyncio.to_thread(self._ccp_context_store.upsert_pack, pack)
            self._register_context_pack(pack)

        return json.dumps({
            "status": "updated_via_recreate",
            "name": name,
            "book": book,
            "size_tokens": pack.size_tokens,
            "message": "Context pack updated by deleting the old entry and creating a replacement.",
        })

    # ── list ───────────────────────────────────────────────────────────────

    def _exec_list_context_packs(
        self: "IkaExecutionEnvironment", _args: dict
    ) -> str:
        book = (_args.get("book") or "").strip() or None
        if self._ccp_context_store is not None:
            packs = []
            for summary in self._ccp_context_store.list_pack_summaries(book):
                name = summary.get("name", "")
                summary_book = summary.get("book_name", self._default_context_book())
                key = self._context_pack_key(name, summary_book)
                existing = self._context_pack_registry.get(key)
                pack = self._ccp_context_store.summary_to_pack(
                    summary,
                    existing_content=existing.content if existing else "",
                )
                key = self._register_context_pack(pack)
                packs.append({
                    "name": name,
                    "book": pack.book_name,
                    "tags": pack.tags,
                    "source": pack.source,
                    "size_tokens": pack.size_tokens,
                    "loaded": key in self._loaded_packs,
                    "created_at": pack.created_at.isoformat(),
                    "updated_at": pack.updated_at.isoformat(),
                })
            return json.dumps({"packs": packs, "total": len(packs), "backend": "ccp"})

        packs = []
        for key, pack in self._context_pack_registry.items():
            if book and pack.book_name != book:
                continue
            packs.append({
                "name": pack.name,
                "book": pack.book_name,
                "tags": pack.tags,
                "source": pack.source,
                "size_tokens": pack.size_tokens,
                "loaded": key in self._loaded_packs,
                "created_at": pack.created_at.isoformat(),
                "updated_at": pack.updated_at.isoformat(),
            })
        return json.dumps({"packs": packs, "total": len(packs)})

    def _exec_search_context_packs(
        self: "IkaExecutionEnvironment", args: dict
    ) -> str:
        query = args.get("query", "").strip()
        book = (args.get("book") or "").strip() or None
        if not query:
            return json.dumps({"error": "query is required"})

        if self._ccp_context_store is not None:
            matches = self._ccp_context_store.search_packs(query, book)
            return json.dumps({"matches": matches, "total": len(matches), "backend": "ccp"})

        matches = []
        query_lower = query.lower()
        for key, pack in self._context_pack_registry.items():
            if book and pack.book_name != book:
                continue
            haystacks = [pack.name, pack.content, pack.source, pack.book_name, *pack.tags]
            if any(query_lower in value.lower() for value in haystacks if isinstance(value, str)):
                matches.append({
                    "name": pack.name,
                    "book": pack.book_name,
                    "tags": pack.tags,
                    "source": pack.source,
                    "size_tokens": pack.size_tokens,
                    "loaded": key in self._loaded_packs,
                })
        return json.dumps({"matches": matches, "total": len(matches), "backend": "local"})
