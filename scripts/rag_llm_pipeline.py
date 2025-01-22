import os
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from transformers import T5ForConditionalGeneration, T5Tokenizer
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

class TinyRAGPipeline:
    def __init__(self,
                 embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
                 llm_model_name="google/flan-t5-base",
                 collection_name="enterprise_chunks",
                 qdrant_url="http://localhost:6333",
                 top_k=3):
        """
        embedding_model_name: E5 veya benzer bir sentence-transformer modeli
        llm_model_name: Flan-T5 küçük model
        collection_name: Qdrant'da hangi koleksiyondan chunk'ları çekeceğiz
        qdrant_url: Qdrant'ın erişim URL'si
        top_k: Kaç chunk geri getirilecek
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Embedding modeli
        print("Loading embedding model...")
        self.emb_model = SentenceTransformer(embedding_model_name, device=self.device)

        # LLM (Flan-T5)
        print("Loading LLM (Flan-T5-Small)...")
        self.llm_tokenizer = T5Tokenizer.from_pretrained(llm_model_name)
        self.llm_model = T5ForConditionalGeneration.from_pretrained(llm_model_name)
        self.llm_model.to(self.device)

        # Qdrant Client
        self.qdrant = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.top_k = top_k

    def embed_query(self, query_text: str):
        """
        E5 modelinde sorguyu 'query: ' prefixiyle encode etme öneriliyor
        (resmi E5 yönergelerine göre).
        """
        text_with_prefix = f"query: {query_text}"
        embedding = self.emb_model.encode([text_with_prefix], convert_to_numpy=True)[0]
        return embedding

    def retrieve_chunks(self, query_embedding):
        """
        Qdrant'tan en benzer top_k chunk'ları döndürür.
        """
        search_result = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=self.top_k
        )
        return search_result

    def generate_answer(self, user_query: str, retrieved_chunks: list):
        """
        Alınan chunk'ları prompt'a ekleyerek Flan-T5 ile yanıt üretir.
        """
        # Context string oluştur
        context_texts = []
        for hit in retrieved_chunks:
            # chunk metnini eğer 'payload' içine kaydetmişsek buradan alabiliriz.
            # Örneğin upsert ederken 'payload': {"chunk_text": ...} gibi eklemiş olabilirsiniz.
            # Bu örnekte, payload'ta chunk_text yoksa, metadata vb. eklemeniz gerekir.
            # Aşağıda, "Maalesef 'chunk_text' yok" diyen senaryoya göre ayarlayabilirsiniz.
            payload = hit.payload
            # E5 pipeline sırasında chunk_text'i kaydettiyseniz:
            # chunk_text = payload.get("chunk_text", "")
            # Bu demo'da chunk_text'i veritabanına koymadıysak, meta veride yoksa:
            # -> Sadece ID, file name gibi alanlar olabilir. 
            # -> RAG pipeline'da chunk_text'i de payload'a koymanız önerilir.
            # (Aşağıdaki satırı istediğiniz gibi uyarlayın.)
            chunk_text = payload.get("text", "")  
            if chunk_text:
                context_texts.append(chunk_text)

        # T5'e verilecek prompt tasarımı
        # Basit bir şablon:
        # "Soru: <user_query>\nBağlam: <chunk1> <chunk2>...\nCevap:"
        context_joined = "\n".join(context_texts)
        prompt = (
            f"Soru: {user_query}\n\n"
            f"Bağlam:\n{context_joined}\n\n"
            f"Lütfen yukarıdaki bağlama dayanarak soruyu yanıtla. "
            f"Eğer yeterli bilgi yoksa 'Bilmiyorum' de.\nCevap:"
        )

        input_ids = self.llm_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        # Generate parametreleri (beam_size, max_length, temperature, vs.) ayarlayabilirsiniz
        outputs = self.llm_model.generate(
            input_ids=input_ids,
            max_length=256,
            num_beams=3,
            early_stopping=True
        )
        answer = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

    def run_qa(self, user_query: str):
        """
        Dışarıdan tek fonksiyonla çağrılacak RAG Q&A akışı.
        """
        # 1) Query embedding
        q_emb = self.embed_query(user_query)
        # 2) Retrieve top_k chunks
        retrieved = self.retrieve_chunks(q_emb)
        # 3) LLM ile cevap
        answer = self.generate_answer(user_query, retrieved)
        return answer


if __name__ == "__main__":
    # Örnek kullanım:
    pipeline = TinyRAGPipeline(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",  # sorgu vektörü
        llm_model_name="google/flan-t5-base",                  # tiny LLM
        collection_name="enterprise_chunks",                    # Qdrant koleksiyon adı
        qdrant_url="http://localhost:6333",
        top_k=3
    )

    while True:
        user_input = input("\nSoru (çıkmak için boş bırak Enter'a bas): ")
        if not user_input.strip():
            print("Çıkış yapılıyor.")
            break
        answer = pipeline.run_qa(user_input)
        print(f"Yanıt: {answer}\n")
