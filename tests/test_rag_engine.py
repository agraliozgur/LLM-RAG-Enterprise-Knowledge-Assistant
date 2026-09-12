import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import rag_engine


class FakeTokenizer:
    model_max_length = 512

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join("token" for _ in ids)


class FakeVector(list):
    def tolist(self):
        return list(self)


def _point(score, collection_name, chunk_id):
    return SimpleNamespace(
        score=score,
        payload={
            "collection_name": collection_name,
            "document_id": f"doc-{chunk_id}",
            "source_file": f"{collection_name}.txt",
            "chunk_id": chunk_id,
            "text": "evidence " * 30,
        },
    )


class RagEngineTests(unittest.TestCase):
    def test_all_collection_search_merges_rankings_and_calls_llm_once(self):
        calls = []

        class Client:
            def query_points(self, collection_name, query, limit, score_threshold):
                calls.append(collection_name)
                points = {
                    "rag_collection_a": [_point(0.60, "a", 1)],
                    "rag_collection_b": [_point(0.90, "b", 2)],
                }[collection_name]
                return SimpleNamespace(points=points)

        llm_calls = []
        with patch.object(rag_engine, "_get_qdrant_client", return_value=Client()), patch.object(
            rag_engine, "_get_embed_model",
            return_value=SimpleNamespace(
                encode=lambda *args, **kwargs: [FakeVector([0.1, 0.2])]
            ),
        ), patch.object(rag_engine, "_get_tokenizer", return_value=FakeTokenizer()), patch.object(
            rag_engine, "_get_llm_pipeline",
            return_value=lambda prompt: llm_calls.append(prompt) or [{"generated_text": "grounded"}],
        ):
            result = rag_engine.answer_with_rag(
                "question", score_threshold=0.45,
                collection_names=["rag_collection_a", "rag_collection_b"],
            )

        self.assertEqual(calls, ["rag_collection_a", "rag_collection_b"])
        self.assertEqual(result["answer"], "grounded")
        self.assertEqual(result["sources"][0]["source_collection"], "b")
        self.assertEqual(len(llm_calls), 1)

    def test_below_threshold_abstains_without_loading_llm(self):
        with patch.object(
            rag_engine, "_get_embed_model",
            return_value=SimpleNamespace(
                encode=lambda *args, **kwargs: [FakeVector([0.1])]
            ),
        ), patch.object(
            rag_engine, "_get_qdrant_client",
            return_value=SimpleNamespace(
                query_points=lambda **kwargs: SimpleNamespace(points=[])
            ),
        ), patch.object(
            rag_engine, "_get_llm_pipeline",
            side_effect=AssertionError("LLM must not be loaded"),
        ):
            result = rag_engine.answer_with_rag(
                "off topic", score_threshold=0.45, collection_names=["rag_collection_a"]
            )
        self.assertEqual(
            result,
            {
                "answer": rag_engine.NOT_FOUND_MSG,
                "sources": [],
                "above_threshold": False,
            },
        )
