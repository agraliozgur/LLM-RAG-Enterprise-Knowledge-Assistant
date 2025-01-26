import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Community-based or older imports for embeddings/vectorstores
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFacePipeline

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# For the RAG chain
from langchain.chains import RetrievalQA

##################################################
# CONFIGURATION
##################################################
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "enterprise_chunks"

LLM_MODEL_NAME = "google/flan-t5-large"
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

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

def main():
    ########################
    # 1) Embeddings & Qdrant
    ########################
    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    # Connect to Qdrant
    qdrant_client = QdrantClient(url=QDRANT_URL)

    # If you haven't already, ensure your Qdrant collection is set up with the correct dimension.
    # We skip the "recreate_collection" step here, assuming it’s done already.

    vectorstore = Qdrant(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embeddings=embedding_model,
    )

    ########################
    # 2) Build RAG Chain
    ########################
    # Create an LLM pipeline for text generation
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)


    print("Loading embedding model...")
    device = get_device()
    hf_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.0,
        device=device
    )
    llm_for_rag = HuggingFacePipeline(pipeline=hf_pipeline)

    # RetrievalQA chain for RAG approach
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_for_rag,
        chain_type="stuff",  # or map_reduce, refine, etc.
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
    )

    ########################
    # 3) Build a Direct LLM Pipeline (No Retrieval)
    ########################
    # This is a raw text2text pipeline for the same model,
    # so we can compare how it answers *without* retrieval context.
    direct_llm_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.0,
        device=device
    )

    ########################
    # 4) Define helper functions
    ########################
    def answer_with_rag(question: str) -> str:
        """Use the RAG pipeline (RetrievalQA) to answer the question."""
        response = qa_chain.invoke(question)  # or .invoke(question)
        return response["result"]

    def answer_with_llm_direct(question: str) -> str:
        """Use the direct pipeline (no retrieval)."""
        # We'll feed the question as is. For T5, you might want to add a prefix like: "Answer the question: ..."
        prompt = f"Answer this question: {question}"
        raw_output = direct_llm_pipeline(prompt)
        # The pipeline returns a list of dicts, e.g. [{"generated_text": "..."}]
        return raw_output[0]["generated_text"]

    ########################
    # 5) Let’s compare answers
    ########################
    while True:
        query = input("\nEnter your question (or 'exit' to quit): ")
        if query.lower() in ["exit", "quit"]:
            print("Exiting.")
            break

        # 5a) LLM alone
        default_answer = answer_with_llm_direct(query)

        # 5b) RAG-based
        rag_answer = answer_with_rag(query)

        # 5c) Print both
        print("\n=== COMPARISON ===")
        print(f"QUESTION: {query}")
        print(f"DEFAULT LLM ANSWER (no retrieval):\n{default_answer}")
        print("-"*40)
        print(f"RAG ANSWER (with retrieval):\n{rag_answer}")
        print("===")

if __name__ == "__main__":
    main()

"""
What is described in our corporate leave policy?
"""