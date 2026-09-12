import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from embed_and_index import _make_point_id, index_chunk_records
from data_preprocessing import clean_text, preprocess_document


class CollectionPipelineTests(unittest.TestCase):
    def test_collection_point_ids_are_stable_per_document_and_chunk(self):
        document_id = "a" * 64
        self.assertEqual(_make_point_id(document_id, 0), _make_point_id(document_id, 0))
        self.assertNotEqual(_make_point_id(document_id, 0), _make_point_id(document_id, 1))
        self.assertNotEqual(_make_point_id(document_id, 0), _make_point_id("b" * 64, 0))

    def test_per_document_preprocessing_writes_collection_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "guide.txt"
            source.write_text("This line is long enough to become a text chunk in the collection pipeline.\n")
            output = root / "chunks.jsonl"
            records = preprocess_document(
                str(source), "default", "a" * 64, str(output)
            )
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["collection_name"], "default")
            self.assertEqual(records[0]["document_id"], "a" * 64)
            self.assertEqual(records[0]["source_file"], "guide.txt")
            self.assertTrue(output.exists())

    def test_clean_text_preserves_french_diacritics(self):
        self.assertIn("français", clean_text("Le français est nécessaire."))
        self.assertIn("élève", clean_text("Un élève apprend."))

    def test_indexing_adds_collection_payload_and_deletes_replaced_document(self):
        class Vector(list):
            def tolist(self):
                return list(self)

        class Matrix(list):
            def tolist(self):
                return [list(row) for row in self]

        class Model:
            def get_sentence_embedding_dimension(self):
                return 2

            def encode(self, texts, convert_to_numpy=True):
                return Matrix([Vector([0.1, 0.2]) for _ in texts])

        class Client:
            def __init__(self):
                self.deleted = []
                self.upserts = []

            def get_collections(self):
                return type("Collections", (), {"collections": []})()

            def create_collection(self, **kwargs):
                self.created = kwargs

            def delete(self, **kwargs):
                self.deleted.append(kwargs)

            def upsert(self, **kwargs):
                self.upserts.append(kwargs)

        records = [{
            "collection_name": "default",
            "document_id": "a" * 64,
            "source_file": "guide.txt",
            "source_path": "data/raw/default/guide.txt",
            "extension": ".txt",
            "chunk_id": 0,
            "text": "chunk text",
        }]
        client = Client()
        index_chunk_records(
            client, Model(), records, "rag_collection_default",
            replace_document_id="b" * 64,
        )
        self.assertEqual(client.created["collection_name"], "rag_collection_default")
        self.assertEqual(len(client.deleted), 1)
        point = client.upserts[0]["points"][0]
        self.assertEqual(point["payload"]["collection_name"], "default")
        self.assertEqual(point["payload"]["document_id"], "a" * 64)
