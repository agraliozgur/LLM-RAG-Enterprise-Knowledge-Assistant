import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from collection_manager import CollectionError, CollectionManager


class CollectionManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.manager = CollectionManager(project_root=self.root)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_create_upload_sync_and_duplicate_rejection(self):
        self.assertTrue(self.manager.create_collection("product-docs"))
        self.assertFalse(self.manager.create_collection("product-docs"))
        self.assertEqual(self.manager.list_collections(), ["product-docs"])

        document = self.manager.upload_bytes(
            "product-docs", "guide.txt", b"useful document"
        )
        self.assertEqual(document["status"], "pending")
        self.assertEqual(
            (self.root / "data/raw/product-docs/guide.txt").read_bytes(),
            b"useful document",
        )
        with self.assertRaisesRegex(CollectionError, "already exists"):
            self.manager.upload_bytes("product-docs", "guide.txt", b"new content")
        with self.assertRaisesRegex(CollectionError, "content already exists"):
            self.manager.upload_bytes("product-docs", "copy.txt", b"useful document")
        with self.assertRaises(CollectionError):
            self.manager.create_collection("Not a slug")

    def test_pending_processing_is_incremental_and_changed_document_replaces_old_id(self):
        self.manager.create_collection("default")
        self.manager.upload_bytes("default", "guide.txt", b"first content")
        index_calls = []

        def preprocess(path, collection_name, document_id, output_path):
            self.assertEqual(Path(path).name, "guide.txt")
            self.assertEqual(collection_name, "default")
            self.assertTrue(output_path.endswith(f"{document_id}.jsonl"))
            return [{
                "collection_name": collection_name,
                "document_id": document_id,
                "source_file": "guide.txt",
                "source_path": "data/raw/default/guide.txt",
                "extension": ".txt",
                "chunk_id": 0,
                "text": "chunk",
            }]

        def index(client, model, records, collection_name, replace_document_id=None):
            index_calls.append((collection_name, records[0]["document_id"], replace_document_id))

        first = self.manager.process_pending(
            "default", qdrant_client=object(), embedding_model=object(),
            preprocess=preprocess, index=index,
        )
        self.assertEqual(first, {"processed": 1, "failed": 0, "skipped": 0})
        self.assertEqual(len(index_calls), 1)
        self.assertEqual(
            self.manager.process_pending(
                "default", qdrant_client=object(), embedding_model=object(),
                preprocess=preprocess, index=index,
            ),
            {"processed": 0, "failed": 0, "skipped": 0},
        )

        old_document_id = index_calls[0][1]
        (self.root / "data/raw/default/guide.txt").write_bytes(b"changed content")
        documents = self.manager.sync_collection("default")
        self.assertEqual(documents[0]["status"], "pending")
        self.assertEqual(documents[0]["indexed_document_id"], old_document_id)
        self.manager.process_pending(
            "default", qdrant_client=object(), embedding_model=object(),
            preprocess=preprocess, index=index,
        )
        self.assertEqual(index_calls[-1][2], old_document_id)

    def test_remove_archives_original_and_deletes_indexed_points(self):
        self.manager.create_collection("default")
        self.manager.upload_bytes("default", "guide.txt", b"content")

        def preprocess(path, collection_name, document_id, output_path):
            return [{
                "collection_name": collection_name, "document_id": document_id,
                "source_file": "guide.txt", "source_path": "data/raw/default/guide.txt",
                "extension": ".txt", "chunk_id": 0, "text": "chunk",
            }]

        self.manager.process_pending(
            "default", qdrant_client=object(), embedding_model=object(),
            preprocess=preprocess, index=lambda *args, **kwargs: None,
        )

        class Client:
            def __init__(self):
                self.deleted = []

            def delete(self, **kwargs):
                self.deleted.append(kwargs)

        client = Client()
        self.assertEqual(self.manager.remove_documents("default", ["guide.txt"], client), 1)
        self.assertFalse((self.root / "data/raw/default/guide.txt").exists())
        self.assertEqual(self.manager.list_documents("default"), [])
        self.assertEqual(len(client.deleted), 1)
        self.assertTrue(list((self.root / "data/cleaned_data/collection_backups").rglob("guide.txt")))

    def test_processing_failure_stays_pending_with_error(self):
        self.manager.create_collection("default")
        self.manager.upload_bytes("default", "bad.txt", b"bad")

        def failing_preprocess(*args):
            raise ValueError("cannot extract")

        summary = self.manager.process_pending(
            "default", qdrant_client=object(), embedding_model=object(),
            preprocess=failing_preprocess, index=lambda *args, **kwargs: None,
        )
        self.assertEqual(summary, {"processed": 0, "failed": 1, "skipped": 0})
        document = self.manager.list_documents("default")[0]
        self.assertEqual(document["status"], "pending")
        self.assertEqual(document["last_error"], "cannot extract")
