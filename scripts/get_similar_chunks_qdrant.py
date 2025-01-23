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
    # user_query_str = "I tried speaking French, as carefully as I could (intermediate level) and it repeatedly transcribed something in Portuguese, which I definitely don't speak. My pronunciation can't be that bad, can it? Kind of discouraging. 1 Reply tsyrak OP 2y ago Releasing the option to turn OFF multilingual mode later today. I can't say about your pronunciation personally. In general though, pronunciation is sooo neglected in language learning it's a shame. Hopefully learning through speech will help a ton. 1 Reply tsyrak OP 2y ago Released the option to turn off multilingual speech recognition. Does it help? 1723\n12325, 12:47 AM I made a chatbot that lets you SPEAK to a French teacher : rlearnfrench Skip to mai1n content Reply Log In ImpossibleFox7622 2y ago Is this using GPT 3.5 or 4? 1 Reply tsyrak OP 2y ago 3.5 1 Reply citizenfaguo 2y ago Looks great. Do you intend to add Chinese soon? 1 Reply tsyrak OP 2y ago Thanks! Yes! A couple of friends have been requesting it, but days are always too short Will try to get this done at last tomorrow! 1 Reply kiva 2y ago Merci!"
    query_similar_chunks(user_query_str)


"""
Search Results:
Score: 0.9203, Chunk ID: 0, Source: task_management_automation.png
Score: 0.8587, Chunk ID: 7, Source: Twitter-Age Knowledge Management for You and Employees.pdf
Score: 0.8560, Chunk ID: 7, Source: How Infosys Built an Enterprise Knowledge Management Assistant Using Generative AI on AWS _ AWS Partner Network (APN) Blog.pdf
"""