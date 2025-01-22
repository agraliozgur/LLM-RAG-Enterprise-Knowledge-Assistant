import os
import json
import uuid
from typing import List
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, CollectionStatus, Query

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

    print("CPU kullanılıyor.")
    return torch.device('cpu')

# 1) Model Yükleme
# Örnek model: sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

print("Loading embedding model...")
device = get_device()
model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

# 2) Qdrant Client Oluşturma (varsayılan localhost:6333)
qdrant_client = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "enterprise_chunks"

def create_collection_if_not_exists(collection_name: str, vector_size: int):
    """
    Qdrant üzerinde bir koleksiyon yoksa oluşturur.
    """
    collections = qdrant_client.get_collections()
    existing_collections = [c.name for c in collections.collections]

    if collection_name not in existing_collections:
        print(f"Collection '{collection_name}' not found. Creating new collection...")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
    else:
        print(f"Collection '{collection_name}' already exists.")

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Bir liste metni verip, embedding modelinden vektör döndürür.
    """
    # E5 modelinde genelde metin önüne 'query: ' veya 'passage: ' eklenir.
    # Ama chunk'ları "passage: " ile işlemek yaygın bir yaklaşım.
    # (Resmi E5 dokümantasyonunda bu prefixlerden bahsediliyor.)
    processed_texts = [f"passage: {t.strip()}" for t in texts]
    embeddings = model.encode(processed_texts, convert_to_numpy=True)
    return embeddings.tolist()

def upload_jsonl_chunks_to_qdrant(jsonl_path: str, collection_name: str):
    """
    JSONL dosyasını okuyup, her chunk'ı embed edip Qdrant'a yazar.
    """
    # Örnek bir batch listesi hazırlayacağız
    batch_texts = []
    batch_ids = []
    batch_metadata = []

    # Aşağıda batch işlem: her N chunk'ta bir Qdrant'a yazma (örneğin 32, 64, 128).
    BATCH_SIZE = 32

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # data içindeki alanlar: unique_id, file_name, extension, chunk_text, chunk_id, vb.
            chunk_text = data.get("chunk_text", "")
            if not chunk_text.strip():
                continue

            # Metadata
            meta = {
                "source_file": data.get("file_name"),
                "extension": data.get("extension"),
                "chunk_id": data.get("chunk_id"),
                "chunk_text": chunk_text
                # gerektiğinde ek alanlar da eklenebilir
            }

            # Qdrant'a gönderilecek ID:
            #   -> Orijinal unique_id + chunk_id kombinasyonundan bir UUID türetebiliriz
            #   -> Veya data içinde halihazırda bir "uuid" varsa onu kullanabiliriz.
            # Burada rastgele bir uuid() oluşturabiliriz.
            point_id = str(uuid.uuid4())

            batch_texts.append(chunk_text)
            batch_ids.append(point_id)
            batch_metadata.append(meta)

            if len(batch_texts) >= BATCH_SIZE:
                # Bu batch'i Qdrant'a gönder
                flush_batch_to_qdrant(batch_texts, batch_ids, batch_metadata, collection_name)
                # Sonra batch listelerini temizleyelim
                batch_texts.clear()
                batch_ids.clear()
                batch_metadata.clear()

        # Döngü bitti, elde kalan batch varsa onu da gönderelim
        if batch_texts:
            flush_batch_to_qdrant(batch_texts, batch_ids, batch_metadata, collection_name)

def flush_batch_to_qdrant(texts: List[str], ids: List[str], metas: List[dict], collection_name: str):
    """
    Embedding hesaplayıp Qdrant'a batch halinde insert/upsert yapar.
    """
    vectors = embed_texts(texts)
    # Qdrant 'points' formatı: 
    #  [
    #    { "id": <str or int>, "vector": [float], "payload": {...} },
    #    ...
    #  ]
    points = []
    for i in range(len(vectors)):
        points.append({
            "id": ids[i],
            "vector": vectors[i],
            "payload": metas[i]
        })

    # Upsert metodu, var olan IDs ile çakışma varsa günceller, yoksa ekler.
    qdrant_client.upsert(
        collection_name=collection_name,
        points=points
    )

def main():
    jsonl_path = "../cleaned_data/all_processed_data.jsonl"  # senin oluşturduğun jsonl path'i
    # Modelin boyutunu öğrenmemiz lazım (embedding dimension).
    # E5-small genelde 512 boyutlu. 
    # (Dilersen model.config veya encode örneğiyle dimension'ı teyit edebilirsin)
    example_vector = model.encode(["passage: test"])[0]
    vector_size = len(example_vector)
    print(f"Detected vector size: {vector_size}")

    # Qdrant Collection oluşturma
    create_collection_if_not_exists(COLLECTION_NAME, vector_size)

    # Verileri JSONL'den Qdrant'a yükle
    upload_jsonl_chunks_to_qdrant(jsonl_path, COLLECTION_NAME)

    # Kontrol amaçlı koleksiyon durumunu yazdır.
    coll_info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
    print("Collection status:", coll_info.status)
    if coll_info.status == CollectionStatus.GREEN:
        print("Collection is ready to serve queries!")

if __name__ == "__main__":
    main()
