"""Persistent filesystem-backed collection management for the Streamlit MVP.

The module intentionally uses only the local filesystem plus Qdrant.  It owns
document lifecycle state while the established preprocessing and embedding
modules continue to own extraction, chunking, and vector indexing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

from utils import _PROJECT_ROOT, get_device, load_config


COLLECTION_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
STATE_VERSION = 1


class CollectionError(ValueError):
    """Raised for a user-correctable collection operation error."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


class CollectionManager:
    """Manage collection directories and their small, durable JSON manifest."""

    def __init__(
        self,
        project_root: Optional[Path] = None,
        collection_prefix: Optional[str] = None,
    ) -> None:
        self.project_root = Path(project_root or _PROJECT_ROOT).resolve()
        config = load_config()
        self.collection_prefix = collection_prefix or config["qdrant"].get(
            "collection_prefix", "rag_collection_"
        )
        self.raw_root = self.project_root / "data" / "raw"
        self.cleaned_root = self.project_root / "data" / "cleaned_data"
        self.state_path = self.cleaned_root / "collection_state.json"
        self.chunks_root = self.cleaned_root / "collection_chunks"

    @staticmethod
    def validate_collection_name(name: str) -> str:
        if not isinstance(name, str) or not COLLECTION_NAME_RE.fullmatch(name):
            raise CollectionError(
                "Collection names must use lowercase letters, numbers, hyphens, "
                "or underscores (1-64 characters)."
            )
        return name

    @staticmethod
    def _safe_filename(filename: str) -> str:
        safe_name = Path(filename).name
        if not safe_name or safe_name in {".", ".."} or safe_name != filename:
            raise CollectionError("The uploaded filename is not valid.")
        return safe_name

    def qdrant_collection_name(self, collection_name: str) -> str:
        return f"{self.collection_prefix}{self.validate_collection_name(collection_name)}"

    def _empty_state(self) -> Dict[str, Any]:
        return {"version": STATE_VERSION, "collections": {}}

    def _load_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return self._empty_state()
        with self.state_path.open("r", encoding="utf-8") as file_handle:
            state = json.load(file_handle)
        if state.get("version") != STATE_VERSION or not isinstance(state.get("collections"), dict):
            raise CollectionError("The collection state file has an unsupported format.")
        return state

    def _write_state(self, state: Dict[str, Any]) -> None:
        self.cleaned_root.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="collection_state_", suffix=".tmp", dir=self.cleaned_root
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file_handle:
                json.dump(state, file_handle, ensure_ascii=False, indent=2, sort_keys=True)
                file_handle.write("\n")
            os.replace(temporary_name, self.state_path)
        except Exception:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
            raise

    def create_collection(self, collection_name: str) -> bool:
        """Create an empty collection directory and manifest entry.

        Returns ``True`` only when the collection is newly created.
        """
        collection_name = self.validate_collection_name(collection_name)
        state = self._load_state()
        if collection_name in state["collections"]:
            return False
        # An on-disk directory may predate the manifest (notably the migrated
        # default corpus).  Register it rather than treating that safe case as
        # an error.
        (self.raw_root / collection_name).mkdir(parents=True, exist_ok=True)
        state["collections"][collection_name] = {
            "qdrant_collection": self.qdrant_collection_name(collection_name),
            "created_at": _now(),
            "documents": {},
        }
        self._write_state(state)
        return True

    def list_collections(self) -> list[str]:
        return sorted(self._load_state()["collections"])

    def discover_collections(self) -> list[str]:
        """Register valid direct children of ``data/raw`` missing from state.

        This makes documents copied into a new knowledge-base folder visible in
        the UI without requiring a separate manual registration step.
        """
        state = self._load_state()
        self.raw_root.mkdir(parents=True, exist_ok=True)
        changed = False
        for directory in self.raw_root.iterdir():
            if not directory.is_dir() or not COLLECTION_NAME_RE.fullmatch(directory.name):
                continue
            if directory.name in state["collections"]:
                continue
            state["collections"][directory.name] = {
                "qdrant_collection": self.qdrant_collection_name(directory.name),
                "created_at": _now(),
                "documents": {},
            }
            changed = True
        if changed:
            self._write_state(state)
        return sorted(state["collections"])

    def _collection_entry(self, state: Dict[str, Any], collection_name: str) -> Dict[str, Any]:
        collection_name = self.validate_collection_name(collection_name)
        try:
            return state["collections"][collection_name]
        except KeyError as error:
            raise CollectionError(f"Collection '{collection_name}' does not exist.") from error

    def upload_bytes(self, collection_name: str, filename: str, content: bytes) -> Dict[str, Any]:
        """Persist one original upload as Pending without processing it."""
        collection_name = self.validate_collection_name(collection_name)
        filename = self._safe_filename(filename)
        if not content:
            raise CollectionError("Cannot upload an empty file.")

        state = self._load_state()
        collection = self._collection_entry(state, collection_name)
        documents = collection["documents"]
        if filename in documents or (self.raw_root / collection_name / filename).exists():
            raise CollectionError(f"A file named '{filename}' already exists in this collection.")

        document_id = _sha256_bytes(content)
        if any(doc.get("document_id") == document_id for doc in documents.values()):
            raise CollectionError("This document content already exists in this collection.")

        target = self.raw_root / collection_name / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as file_handle:
            file_handle.write(content)

        document = self._pending_document(collection_name, filename, document_id)
        documents[filename] = document
        self._write_state(state)
        return document.copy()

    def _pending_document(
        self,
        collection_name: str,
        filename: str,
        document_id: str,
        indexed_document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "document_id": document_id,
            "extension": Path(filename).suffix.lower(),
            "status": "pending",
            "indexed_document_id": indexed_document_id,
            "chunk_file": (
                f"collection_chunks/{collection_name}/{document_id}.jsonl"
            ),
            "chunk_count": 0,
            "processed_at": None,
            "indexed_at": None,
            "last_error": None,
        }

    def sync_collection(self, collection_name: str) -> list[Dict[str, Any]]:
        """Register manually present direct files and detect changed bytes.

        Only direct children of ``data/raw/<collection_name>/`` are considered;
        nested folders are intentionally ignored by the collection MVP.
        """
        collection_name = self.validate_collection_name(collection_name)
        state = self._load_state()
        collection = self._collection_entry(state, collection_name)
        collection_dir = self.raw_root / collection_name
        documents = collection["documents"]
        changed = False

        for source_path in sorted(collection_dir.iterdir(), key=lambda path: path.name):
            if not source_path.is_file():
                continue
            filename = source_path.name
            document_id = _sha256_file(source_path)
            existing = documents.get(filename)
            if existing and existing.get("document_id") == document_id:
                continue
            documents[filename] = self._pending_document(
                collection_name,
                filename,
                document_id,
                indexed_document_id=(existing or {}).get("indexed_document_id"),
            )
            changed = True

        if changed:
            self._write_state(state)
        return self.list_documents(collection_name)

    def list_documents(self, collection_name: str) -> list[Dict[str, Any]]:
        state = self._load_state()
        collection = self._collection_entry(state, collection_name)
        return [
            {"filename": filename, **document}
            for filename, document in sorted(collection["documents"].items())
        ]

    def qdrant_names(self, collection_names: Optional[Iterable[str]] = None) -> list[str]:
        state = self._load_state()
        names = collection_names if collection_names is not None else state["collections"].keys()
        result = []
        for name in names:
            collection = self._collection_entry(state, name)
            result.append(collection["qdrant_collection"])
        return result

    def searchable_qdrant_names(
        self, collection_names: Optional[Iterable[str]] = None
    ) -> list[str]:
        """Return indexed collections that also exist in the live Qdrant server."""
        return self.qdrant_health(collection_names)["available"]

    def qdrant_health(
        self, collection_names: Optional[Iterable[str]] = None
    ) -> Dict[str, Any]:
        """Reconcile manifest state with live Qdrant state before retrieval."""
        state = self._load_state()
        names = collection_names if collection_names is not None else state["collections"].keys()
        wanted = []
        for name in names:
            collection = self._collection_entry(state, name)
            if any(
                document.get("status") == "processed"
                for document in collection["documents"].values()
            ):
                wanted.append((name, collection["qdrant_collection"]))
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(url=load_config()["qdrant"]["url"], check_compatibility=False)
            live_names = {entry.name for entry in client.get_collections().collections}
        except Exception as error:
            return {
                "online": False,
                "available": [],
                "missing": [],
                "error": str(error),
            }
        available = [qdrant_name for _, qdrant_name in wanted if qdrant_name in live_names]
        missing = [name for name, qdrant_name in wanted if qdrant_name not in live_names]
        return {"online": True, "available": available, "missing": missing, "error": None}

    def mark_for_reindex(self, collection_name: str, reason: str = "Search index needs rebuilding.") -> int:
        """Safely queue all documents in a knowledge base for a fresh index."""
        collection_name = self.validate_collection_name(collection_name)
        state = self._load_state()
        collection = self._collection_entry(state, collection_name)
        count = 0
        for document in collection["documents"].values():
            if document.get("status") == "processed":
                document["status"] = "pending"
                # The collection is absent, so replacement must not try to
                # delete a non-existent former point set.
                document["indexed_document_id"] = None
                document["last_error"] = reason
                count += 1
        if count:
            self._write_state(state)
        return count

    def remove_documents(
        self,
        collection_name: str,
        filenames: Iterable[str],
        qdrant_client: Any = None,
    ) -> int:
        """Remove documents from search and archive their originals recoverably.

        Originals and chunk artifacts are copied to a timestamped local backup
        before their live copies are removed.  Indexed documents are deleted
        from their Qdrant collection first; if Qdrant is unavailable, nothing
        is removed from the live collection state.
        """
        collection_name = self.validate_collection_name(collection_name)
        selected = sorted(set(filenames))
        if not selected:
            return 0
        state = self._load_state()
        collection = self._collection_entry(state, collection_name)
        documents = collection["documents"]
        unknown = [filename for filename in selected if filename not in documents]
        if unknown:
            raise CollectionError("One or more selected documents no longer exist.")

        selected_documents = [(filename, documents[filename]) for filename in selected]
        indexed = [document for _, document in selected_documents if document.get("indexed_document_id")]
        if indexed and qdrant_client is None:
            from qdrant_client import QdrantClient
            config = load_config()
            qdrant_client = QdrantClient(
                url=config["qdrant"]["url"], check_compatibility=False
            )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive_root = self.cleaned_root / "collection_backups" / timestamp / collection_name
        archive_root.mkdir(parents=True, exist_ok=True)
        for filename, document in selected_documents:
            source_path = self.raw_root / collection_name / filename
            if source_path.exists():
                shutil.copy2(source_path, archive_root / filename)
            chunk_path = self.cleaned_root / document["chunk_file"]
            if chunk_path.exists():
                shutil.copy2(chunk_path, archive_root / chunk_path.name)

        if indexed:
            from embed_and_index import delete_document_points
            for document in indexed:
                delete_document_points(
                    qdrant_client,
                    collection["qdrant_collection"],
                    document["indexed_document_id"],
                )

        for filename, document in selected_documents:
            source_path = self.raw_root / collection_name / filename
            if source_path.exists():
                source_path.unlink()
            chunk_path = self.cleaned_root / document["chunk_file"]
            if chunk_path.exists():
                chunk_path.unlink()
            documents.pop(filename)
        self._write_state(state)
        return len(selected)

    def process_pending(
        self,
        collection_name: str,
        qdrant_client: Any = None,
        embedding_model: Any = None,
        preprocess: Optional[Callable[..., list]] = None,
        index: Optional[Callable[..., None]] = None,
    ) -> Dict[str, int]:
        """Preprocess and index only Pending documents in one collection."""
        collection_name = self.validate_collection_name(collection_name)
        self.sync_collection(collection_name)
        state = self._load_state()
        collection = self._collection_entry(state, collection_name)
        documents = collection["documents"]
        pending_names = [
            filename for filename, document in documents.items()
            if document.get("status") == "pending"
        ]
        summary = {"processed": 0, "failed": 0, "skipped": 0}
        if not pending_names:
            return summary

        if preprocess is None:
            from data_preprocessing import preprocess_document
            preprocess = preprocess_document
        if index is None or qdrant_client is None or embedding_model is None:
            from embed_and_index import index_chunk_records, load_embedding_model
            from qdrant_client import QdrantClient

            config = load_config()
            index = index or index_chunk_records
            qdrant_client = qdrant_client or QdrantClient(
                url=config["qdrant"]["url"], check_compatibility=False
            )
            embedding_model = embedding_model or load_embedding_model(
                config["models"]["embedding_model_name"],
                device=get_device(),
            )

        for filename in pending_names:
            document = documents[filename]
            try:
                source_path = self.raw_root / collection_name / filename
                if not source_path.is_file():
                    raise FileNotFoundError(f"Original file is missing: {filename}")
                current_id = _sha256_file(source_path)
                if current_id != document["document_id"]:
                    document = self._pending_document(
                        collection_name,
                        filename,
                        current_id,
                        indexed_document_id=document.get("indexed_document_id"),
                    )
                    documents[filename] = document

                chunk_path = self.cleaned_root / document["chunk_file"]
                records = preprocess(
                    str(source_path), collection_name, document["document_id"], str(chunk_path)
                )
                index(
                    qdrant_client,
                    embedding_model,
                    records,
                    collection["qdrant_collection"],
                    replace_document_id=document.get("indexed_document_id"),
                )
                document.update(
                    {
                        "status": "processed",
                        "indexed_document_id": document["document_id"],
                        "chunk_count": len(records),
                        "processed_at": _now(),
                        "indexed_at": _now(),
                        "last_error": None,
                    }
                )
                summary["processed"] += 1
            except Exception as error:  # keep Pending for explicit retry
                document["status"] = "pending"
                document["last_error"] = str(error)
                summary["failed"] += 1
            self._write_state(state)
        return summary
