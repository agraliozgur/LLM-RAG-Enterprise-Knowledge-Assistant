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
from langchain.prompts import PromptTemplate  # <--- We'll use this for custom instructions

##################################################
# CONFIGURATION
##################################################
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "enterprise_chunks"

LLM_MODEL_NAME = "google/flan-t5-large"
# EMBED_MODEL_NAME can only have ONE value at a time, so pick one
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# If you want to switch to e5, comment the above and uncomment:
# EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"

def get_device():
    """Detect GPU, MPS (Apple Silicon), or fallback to CPU."""
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
        print("MPS is not available.")
    except Exception as e:
        print(f"MPS checking: {e}")

    print("CPU is in use.")
    return torch.device('cpu')

def main():
    ########################
    # 1) Embeddings & Qdrant
    ########################
    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    # Connect to Qdrant
    qdrant_client = QdrantClient(url=QDRANT_URL)

    # We assume the Qdrant collection is already created with the correct embedding dimension

    vectorstore = Qdrant(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embeddings=embedding_model,
    )

    ########################
    # 2) Build RAG Chain
    ########################
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)

    print("Loading LLM model/pipeline...")
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

    # --- Create a custom PromptTemplate for RAG
    prompt_template = """
You are an advanced enterprise knowledge assistant with direct access to a vector database of documents.
For each user query, you retrieve the most relevant text passage(s) from the database, along with their similarity scores.

Your task is to produce a clear, correct, and complete answer in English, guided by these rules:

1. If the highest similarity score to the user’s query is below 0.4, inform the user that:
   - “No sufficiently relevant data was found in the database.”
   - If you still have knowledge of the answer outside the database, provide it but explicitly warn:
     “This answer is not from the official database; it is based on general knowledge.”

2. If the highest similarity score is 0.4 or above, incorporate the provided database context to write a coherent answer.
   - Summarize or paraphrase the retrieved text accurately.
   - If you find multiple relevant passages, integrate them into a single, well-structured response.
   - Use professional and concise language.

3. If you are genuinely unsure or the context is incomplete, say: 
   “I’m not sure based on the provided information.”

4. Do not fabricate references or nonexistent data. 
   - It is better to say you do not have the information than to make it up.

Remember, your final output should be a concise paragraph (or a few paragraphs if needed), using correct grammar and spelling. Always ensure the user is aware of whether or not the information came directly from the database.

----

Below is the context retrieved from the database (if any):
{context}

Question:
{question}

Now draft your best possible answer, following the above instructions.

""".strip()

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_for_rag,
        chain_type="stuff",  # 'map_reduce', 'refine', etc. are also possible
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},  # <--- Insert custom prompt
    )

    ########################
    # 3) Build a Direct LLM Pipeline (No Retrieval)
    ########################
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
        response = qa_chain.invoke(question)
        return response["result"]

    def answer_with_llm_direct(question: str) -> str:
        """
        Use the direct pipeline (no retrieval).
        Give a more complete prompt to guide T5 toward a helpful style:
        """
        direct_prompt = f"""
You are an AI assistant specialized in enterprise knowledge. Answer the following question thoroughly.
If you are not sure, say "I don't have enough information."

Question: {question}

Answer:
"""
        raw_output = direct_llm_pipeline(direct_prompt.strip())
        return raw_output[0]["generated_text"]

    ########################
    # 5) Compare answers
    ########################
    while True:
        query = input("\nEnter your question (or 'exit'/'quit' to stop): ")
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
        print(f"\n[DEFAULT LLM ANSWER (no retrieval)]:\n{default_answer}")
        print("-"*40)
        print(f"[RAG ANSWER (with retrieval)]:\n{rag_answer}")
        print("===")

if __name__ == "__main__":
    main()
