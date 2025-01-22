import os
import json
import uuid
from typing import List
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, CollectionStatus


def get_device():
    try:
        if torch.cuda.is_available():
            print("CUDA is using.")
            return torch.device('cuda')
    except Exception as e:
        print(f"CUDA checking: {e}")

    try:
        if torch.backends.mps.is_available():
            print("MPS is using.")
            return torch.device('mps')
    except AttributeError:
        print("MPS desteği mevcut değil.")
    except Exception as e:
        print(f"MPS checking: {e}")

    print("CPU ")
    return torch.device('cpu')

# 1) Model Yükleme
# Örnek model: intfloat/multilingual-e5-small
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

print("Loading embedding model...")
device = device = get_device()
model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

# 2) Qdrant Client Oluşturma (varsayılan localhost:6333)
qdrant_client = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "enterprise_chunks"


def query_similar_chunks(user_query: str, top_k: int = 3):
    # E5 modelinde sorguyu "query: " prefix'i ile encode edebiliriz
    query_embedding = model.encode([f"query: {user_query}"], convert_to_numpy=True)[0].tolist()

    search_result = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k
    )

    print("Search Results:")
    for hit in search_result:
        score = hit.score
        payload = hit.payload
        print(f"Score: {score:.4f}, Chunk ID: {payload.get('chunk_id')}, Source: {payload.get('source_file')}")
        # Qdrant'a chunk metnini de payload olarak kaydetmek istersen, metas'a ekleyebilirsin.

# Örnek kullanım:
if __name__ == "__main__":
    user_query_str = "Knowledge Assistants Live Stream on Jan 23rd:HOME < SOLUTIONS < ENTERPRİSE KNOWLEDGE ASSISTANT Selected Use Cases - Task Management Automation 1. Time-Off Coordination. Al Assistant seamlessiy communicates wit"
    query_similar_chunks(user_query_str)
