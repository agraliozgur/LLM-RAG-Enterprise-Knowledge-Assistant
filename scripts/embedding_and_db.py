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

    print("CPU kullanılıyor.")
    return torch.device('cpu')

# 1) Model Yükleme
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

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
    try:
        collection_info = qdrant_client.get_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' zaten mevcut.")
    except Exception as e:
        print(f"Collection '{collection_name}' bulunamadı. Oluşturuluyor...")
        try:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"Collection '{collection_name}' oluşturuldu.")
        except Exception as ce:
            print(f"Koleksiyon oluşturulurken bir hata oluştu: {ce}")

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Bir liste metni verip, embedding modelinden vektör döndürür.
    """
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

    if not os.path.exists(jsonl_path):
        print(f"JSONL dosyası bulunamadı: {jsonl_path}")
        return

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            try:
                data = json.loads(line)
            except json.JSONDecodeError as jde:
                print(f"JSONDecodeError satır {line_number}: {jde}")
                continue

            # data içindeki alanlar: unique_id, file_name, extension, chunk_text, chunk_id, vb.
            chunk_text = data.get("chunk_text", "")
            if not chunk_text.strip():
                print(f"Satır {line_number} boş chunk_text içeriyor. Atlanıyor.")
                continue

            # Metadata
            meta = {
                "source_file": data.get("file_name"),
                "extension": data.get("extension"),
                "chunk_id": data.get("chunk_id"),
                "text":data.get("chunk_text"),
                # gerektiğinde ek alanlar da eklenebilir
            }

            # Qdrant'a gönderilecek ID:
            # Rastgele bir uuid oluşturuyoruz.
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

    try:
        # Upsert metodu, var olan IDs ile çakışma varsa günceller, yoksa ekler.
        response = qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"{len(points)} doküman koleksiyona eklendi/upsert edildi.")
    except Exception as e:
        print(f"Upsert işlemi sırasında bir hata oluştu: {e}")

def query_similar_chunks(user_query: str, top_k: int = 3):
    """
    Kullanıcı sorgusuna benzer chunk'ları Qdrant'tan arar ve sonuçları yazdırır.
    """
    try:
        # Sorguyu encode et
        query_embedding = model.encode([f"query: {user_query}"], convert_to_numpy=True)[0].tolist()

        # search metodunu kullanarak arama yap
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True  # Payload'ı almak için
        )

        print("Search Results:")
        for hit in search_result:
            score = hit.score
            payload = hit.payload
            print(f"Score: {score:.4f}, Chunk ID: {payload.get('chunk_id')}, Source: {payload.get('source_file')}")
    except Exception as e:
        print(f"Arama sırasında bir hata oluştu: {e}")

def main():
    jsonl_path = "../cleaned_data/all_processed_data.jsonl"  # JSONL dosyanızın doğru yolunu kontrol edin
    # Modelin boyutunu öğrenmemiz lazım (embedding dimension).
    example_vector = model.encode(["passage: test"])[0]
    vector_size = len(example_vector)
    print(f"Detected vector size: {vector_size}")

    # Qdrant Collection oluşturma
    create_collection_if_not_exists(COLLECTION_NAME, vector_size)

    # Verileri JSONL'den Qdrant'a yükle
    upload_jsonl_chunks_to_qdrant(jsonl_path, COLLECTION_NAME)

    # Kontrol amaçlı koleksiyon durumunu yazdır.
    try:
        coll_info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        print("Collection status:", coll_info.status)
        if coll_info.status == CollectionStatus.GREEN:
            print("Collection is ready to serve queries!")
    except Exception as e:
        print(f"Collection bilgisi alınırken bir hata oluştu: {e}")

    # Verilerin eklenip eklenmediğini kontrol etmek için sorgu yap
    user_query_str = "CrateDB Blog | Core Technique"
    query_similar_chunks(user_query_str)

if __name__ == "__main__":
    main()
