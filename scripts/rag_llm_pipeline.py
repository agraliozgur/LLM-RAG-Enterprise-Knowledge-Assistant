import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, CollectionStatus
import textwrap

###############################################################################
# 0) Genel Ayarlar
###############################################################################
# Daha önce embedding_and_db.py ile oluşturduğun Qdrant koleksiyon adı
COLLECTION_NAME = "enterprise_chunks"

# E5 embedding modeli
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

# Küçük (tiny) LLM olarak Flan-T5-Small
LLM_MODEL_NAME = "google/flan-t5-large"

# Sorguda en fazla kaç chunk döndürelim?
TOP_K = 3

###############################################################################
# 1) Cihaz (CPU / CUDA / MPS) seçimi
###############################################################################
def get_device():
    """Daha önceki kodunda da benzer şekilde cihaz kontrolü."""
    try:
        if torch.cuda.is_available():
            print("CUDA kullanılıyor.")
            return torch.device('cuda')
    except Exception as e:
        print(f"CUDA kontrol hatası: {e}")

    try:
        if torch.backends.mps.is_available():
            print("MPS (Apple Silicon) kullanılıyor.")
            return torch.device('mps')
    except AttributeError:
        print("MPS desteği mevcut değil.")
    except Exception as e:
        print(f"MPS kontrol hatası: {e}")

    print("CPU kullanılıyor.")
    return torch.device('cpu')

###############################################################################
# 2) RAG Pipeline Sınıfı
###############################################################################
class RAGPipeline:
    def __init__(
        self,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        llm_model_name=LLM_MODEL_NAME,
        collection_name=COLLECTION_NAME,
        top_k=TOP_K,
        qdrant_url="http://localhost:6333"
    ):
        """
        RAG Pipeline kurulumunu yapan sınıf:
          - Embedding modeli (E5)
          - LLM (Flan-T5-Small)
          - Qdrant erişimi
        """
        self.device = get_device()
        self.collection_name = collection_name
        self.top_k = top_k

        print("\n[INFO] Embedding modeli yükleniyor:", embedding_model_name)
        self.embedding_model = SentenceTransformer(embedding_model_name, device=str(self.device))

        print("[INFO] LLM yükleniyor:", llm_model_name)
        self.llm_tokenizer = T5Tokenizer.from_pretrained(llm_model_name)
        self.llm_model = T5ForConditionalGeneration.from_pretrained(llm_model_name).to(self.device)

        print("[INFO] Qdrant client başlatılıyor...")
        self.qdrant_client = QdrantClient(url=qdrant_url)

        # Embedding boyutunu teyit etmek istersen:
        test_emb = self.embedding_model.encode(["passage: test"])[0]
        print(f"[INFO] Embedding dimension: {len(test_emb)}")

        # Koleksiyonun gerçekten mevcut olduğunu kontrol edelim
        collections = self.qdrant_client.get_collections()
        existing = [col.name for col in collections.collections]
        if collection_name not in existing:
            raise ValueError(f"Qdrant koleksiyonu '{collection_name}' bulunamadı! "
                             "Önce embedding_and_db.py ile oluşturduğundan emin olun.")

    def embed_query(self, query: str):
        """E5 modeline uygun prefix 'query:' ekleyerek embedding alır."""
        text_with_prefix = f"query: {query}"
        emb = self.embedding_model.encode([text_with_prefix], convert_to_numpy=True)[0]
        return emb

    def retrieve_relevant_chunks(self, query_embedding):
        """
        Qdrant'tan top_k chunk'ları arar.
        """
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=self.top_k
        )
        return results

    def generate_llm_answer(self, prompt: str, max_tokens=256, num_beams=3):
        """
        Flan-T5 modeliyle prompt'a yanıt üretir.
        """
        input_ids = self.llm_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        outputs = self.llm_model.generate(
            input_ids=input_ids,
            max_length=max_tokens,
            num_beams=num_beams,
            early_stopping=True
        )
        answer = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

    def answer_with_rag(self, user_query: str):
        """
        Kullanıcının sorusuna RAG yaklaşımıyla yanıt döndürür.
        1) Sorgu embedding
        2) Qdrant'ta benzer chunk'lar
        3) Chunk'ları prompt'a ekle + LLM'e ver
        """

        # 1) Sorguyu embed et
        query_emb = self.embed_query(user_query)

        # 2) Qdrant'ta chunk arama
        search_results = self.retrieve_relevant_chunks(query_emb)

        # 3) Seçilen chunk'lardan bir "context" metni oluştur
        #    embedding_and_db.py dosyasında "payload" içinde "chunk_text" saklamıştık.
        context_texts = []
        for r in search_results:
            payload = r.payload

            chunk_text = payload.get("chunk_text", "")
            # print(f"000****   T'{chunk_text}'T *****00000")
            if chunk_text.strip():
                context_texts.append(chunk_text)

        # Bağlam metnini tek bir string halinde birleştiriyoruz
        context_joined = "\n---\n".join(context_texts)

        # 4) Prompt oluştur
        # basit bir şablon: 
        # "Soru: <user_query>\nBağlam: <context_joined>\nCevap:"
        prompt = (
            f"Soru: {user_query}\n\n"
            f"Bağlam:\n{context_joined}\n\n"
            f"Lütfen yukarıdaki bağlama dayanarak soruyu yanıtla. "
            f"Eğer bilgiden emin değilsen 'Bilmiyorum' de.\nCevap:"
        )

        # 5) LLM cevabı
        rag_answer = self.generate_llm_answer(prompt)
        return rag_answer

    def answer_default_llm(self, user_query: str):
        """
        Hiçbir bağlam kullanmadan, sadece soruyu LLM'e verip cevap alır.
        """
        prompt = f"Soru: {user_query}\nCevap:"
        return self.generate_llm_answer(prompt)

###############################################################################
# 3) Ana Çalışma Bloğu (Interactive)
###############################################################################
if __name__ == "__main__":
    print("=== RAG Pipeline Başlatılıyor ===")
    pipeline = RAGPipeline(
        embedding_model_name=EMBEDDING_MODEL_NAME,
        llm_model_name=LLM_MODEL_NAME,
        collection_name=COLLECTION_NAME,
        top_k=TOP_K,
        qdrant_url="http://localhost:6333"  # Qdrant'ın URL'i
    )

    print("\nSoru girerek, hem 'Default LLM' hem 'RAG' cevabını görebilirsiniz.")
    print("Çıkmak için boş bırakıp Enter'a basın.\n")

    while True:
        user_query = input("Sorunuz: ").strip()
        if not user_query:
            print("Çıkış yapılıyor.")
            break

        # 1) Default LLM cevabı
        default_answer = pipeline.answer_default_llm(user_query)

        # 2) RAG cevabı
        rag_answer = pipeline.answer_with_rag(user_query)

        # 3) Ekranda göster
        print("\n[LLM (default) cevabı]:")
        print(textwrap.fill(default_answer, width=80))

        print("\n[LLM (RAG) cevabı]:")
        print(textwrap.fill(rag_answer, width=80))

        print("\n" + "="*80 + "\n")
