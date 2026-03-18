"""CCP-backed storage adapter for IkaGeneral context packs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from .helpers import estimate_tokens
from .models import CcpContextConfig, ContextPack

_METADATA_KIND = "ika_general_context_pack"


class CcpContextStore:
    """Persist and retrieve context packs from a CCP session."""

    def __init__(self, config: CcpContextConfig):
        self._config = config
        self._client_binary = self._resolve_client_binary(config.client_binary)

    @property
    def session(self) -> str:
        return self._config.session

    @property
    def shelf_name(self) -> str:
        return self._config.shelf_name

    @property
    def book_name(self) -> str:
        return self._config.book_name

    def ensure_namespace(self, book_name: str | None = None) -> str:
        target_book = book_name or self.book_name
        shelves = self._run_json(["search-shelves", self.session, self.shelf_name])
        if not any(item.get("shelf_name") == self.shelf_name for item in shelves):
            self._run_json(["add-shelf", self.session, self.shelf_name, self._config.shelf_description])

        books = self._run_json(["search-books", self.session, target_book])
        if not any(
            item.get("shelf_name") == self.shelf_name and item.get("book_name") == target_book
            for item in books
        ):
            self._run_json(
                [
                    "add-book",
                    self.session,
                    "--shelf",
                    self.shelf_name,
                    target_book,
                    self._config.book_description,
                ]
            )
        return target_book

    def list_pack_summaries(self, book_name: str | None = None) -> list[dict[str, Any]]:
        target_book = self.ensure_namespace(book_name)
        entries = self._run_json(["list", self.session])
        return [
            entry for entry in entries
            if entry.get("shelf_name") == self.shelf_name
            and (book_name is None or entry.get("book_name") == target_book)
        ]

    def get_pack(self, name: str, book_name: str | None = None) -> ContextPack:
        target_book = self.ensure_namespace(book_name)
        entry = self._run_json(
            ["get", self.session, name, "--shelf", self.shelf_name, "--book", target_book]
        )
        return self._entry_to_pack(entry)

    def delete_pack(self, name: str, book_name: str | None = None) -> bool:
        target_book = self.ensure_namespace(book_name)
        if not self.has_pack(name, target_book):
            return False
        self._run_json(
            ["delete", self.session, name, "--shelf", self.shelf_name, "--book", target_book]
        )
        return True

    def upsert_pack(self, pack: ContextPack) -> ContextPack:
        target_book = self.ensure_namespace(pack.book_name)
        if self.has_pack(pack.name, target_book):
            self._run_json(
                ["delete", self.session, pack.name, "--shelf", self.shelf_name, "--book", target_book]
            )

        cmd = [
            "add-entry",
            self.session,
            "--shelf",
            self.shelf_name,
            "--book",
            target_book,
            pack.name,
            self._serialize_metadata(pack),
        ]
        if pack.tags:
            cmd.extend(["--labels", ",".join(pack.tags)])
        cmd.append(pack.content)
        entry = self._run_json(cmd)
        return self._entry_to_pack(entry)

    def search_packs(self, query: str, book_name: str | None = None) -> list[dict[str, Any]]:
        target_book = self.ensure_namespace(book_name)
        entries = self._run_json(["search-entries", self.session, query])
        matches = [
            entry for entry in entries
            if entry.get("shelf_name") == self.shelf_name
            and (book_name is None or entry.get("book_name") == target_book)
        ]
        results: list[dict[str, Any]] = []
        for entry in matches:
            metadata = self._parse_metadata(entry.get("description", ""))
            results.append({
                "name": entry.get("name", ""),
                "description": metadata.get("description", entry.get("description", "")),
                "tags": entry.get("labels", []),
                "source": metadata.get("source", "ccp"),
                "shelf_name": entry.get("shelf_name", self.shelf_name),
                "book_name": entry.get("book_name", self.book_name),
            })
        return results

    def has_pack(self, name: str, book_name: str | None = None) -> bool:
        return any(entry.get("name") == name for entry in self.list_pack_summaries(book_name))

    def _serialize_metadata(self, pack: ContextPack) -> str:
        payload = {
            "kind": _METADATA_KIND,
            "source": pack.source,
            "created_at": pack.created_at.isoformat(),
            "updated_at": pack.updated_at.isoformat(),
            "size_tokens": pack.size_tokens or estimate_tokens(pack.content),
            "description": f"IkaGeneral context pack '{pack.name}'",
        }
        return json.dumps(payload, separators=(",", ":"))

    def _entry_to_pack(self, entry: dict[str, Any]) -> ContextPack:
        metadata = self._parse_metadata(entry.get("description", ""))
        created_at = self._parse_datetime(metadata.get("created_at"))
        updated_at = self._parse_datetime(metadata.get("updated_at")) or datetime.now()
        return ContextPack(
            name=entry.get("name", ""),
            content=entry.get("context", ""),
            book_name=entry.get("book_name", self.book_name),
            tags=list(entry.get("labels", [])),
            source=metadata.get("source", "ccp"),
            created_at=created_at or updated_at,
            updated_at=updated_at,
            size_tokens=int(metadata.get("size_tokens") or estimate_tokens(entry.get("context", ""))),
        )

    def summary_to_pack(self, entry: dict[str, Any], existing_content: str = "") -> ContextPack:
        metadata = self._parse_metadata(entry.get("description", ""))
        created_at = self._parse_datetime(metadata.get("created_at")) or datetime.now()
        updated_at = self._parse_datetime(metadata.get("updated_at")) or created_at
        return ContextPack(
            name=entry.get("name", ""),
            content=existing_content,
            book_name=entry.get("book_name", self.book_name),
            tags=list(entry.get("labels", [])),
            source=metadata.get("source", "ccp"),
            created_at=created_at,
            updated_at=updated_at,
            size_tokens=int(metadata.get("size_tokens") or estimate_tokens(existing_content)),
        )

    def _run_json(self, args: list[str]) -> Any:
        env = os.environ.copy()
        if self._config.client_home:
            env["CCP_CLIENT_HOME"] = self._config.client_home

        proc = subprocess.run(
            [self._client_binary, *args],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip() or f"exit code {proc.returncode}"
            raise RuntimeError(f"CCP command failed: {' '.join(args)}: {detail}")

        stdout = proc.stdout.strip()
        if not stdout:
            return None
        try:
            return json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"CCP command returned non-JSON output for {' '.join(args)}: {stdout}"
            ) from exc

    def _resolve_client_binary(self, configured_binary: str | None) -> str:
        candidates = [
            configured_binary,
            os.getenv("CCP_CLIENT_BIN"),
            shutil.which("client"),
            str(Path.home() / "Cephalopod-Coordination-Protocol" / "src" / "client" / "target" / "release" / "client"),
        ]
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return candidate
        raise FileNotFoundError(
            "Unable to locate the CCP client binary. Set CCP_CLIENT_BIN or pass client_binary in ccp_context()."
        )

    @staticmethod
    def _parse_metadata(raw: str) -> dict[str, Any]:
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"description": raw}
        if isinstance(parsed, dict):
            return parsed
        return {"description": raw}

    @staticmethod
    def _parse_datetime(raw: Any) -> datetime | None:
        if not raw or not isinstance(raw, str):
            return None
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None
