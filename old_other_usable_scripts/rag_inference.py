import logging
import sys

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Community-based or specialized imports for embeddings/vector stores
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFacePipeline

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams  # noqa: F401  # (In case needed for expansions)

# For RetrievalQA
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from alignment import align_answer


# --------------------------------------------------
# Logging Configuration
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Global Configuration
# --------------------------------------------------
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "enterprise_chunks"

LLM_MODEL_NAME = "google/flan-t5-large"

# Only one embed model can be used at a time; pick the one you prefer:
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Uncomment if you want to switch to E5-based embeddings:
# EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"

# --------------------------------------------------
# Device Detection
# --------------------------------------------------
def get_device() -> torch.device:
    """
    Detect GPU, MPS (Apple Silicon), or fallback to CPU.

    Returns:
        torch.device: The best available torch device.
    """
    # Try CUDA
    try:
        if torch.cuda.is_available():
            logger.info("Using CUDA (GPU).")
            return torch.device("cuda")
    except Exception as exc:
        logger.warning(f"CUDA check failed: {exc}")

    # Try MPS (Apple Silicon)
    try:
        if torch.backends.mps.is_available():
            logger.info("Using MPS (Apple Silicon).")
            return torch.device("mps")
    except AttributeError:
        logger.warning("MPS is not supported in this environment.")
    except Exception as exc:
        logger.warning(f"MPS check failed: {exc}")

    # Fallback to CPU
    logger.info("Using CPU.")
    return torch.device("cpu")

# --------------------------------------------------
# Main Routine
# --------------------------------------------------
def main() -> None:
    """
    1) Load embeddings & connect to Qdrant.
    2) Build a RetrievalQA (RAG) chain with a custom prompt.
    3) Build a direct LLM pipeline without retrieval.
    4) Prompt user for questions and compare answers from:
        (a) Plain LLM 
        (b) RAG-based approach
    """
    # 1) Embeddings & Qdrant
    logger.info("Initializing embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    logger.info(f"Connecting to Qdrant at: {QDRANT_URL}")
    qdrant_client = QdrantClient(url=QDRANT_URL)

    # Assumes the Qdrant collection already exists with correct embedding dimension
    logger.info(f"Setting up Qdrant vector store for collection: '{COLLECTION_NAME}'")
    vectorstore = Qdrant(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embeddings=embedding_model,
    )

    # 2) Build RAG Chain
    logger.info(f"Loading LLM model '{LLM_MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)

    device = get_device()
    hf_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.0,
        device=device
    )

    # Wrap HF pipeline for LangChain
    llm_for_rag = HuggingFacePipeline(pipeline=hf_pipeline)

    prompt_template_text = """
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

    prompt = PromptTemplate(
        template=prompt_template_text,
        input_variables=["context", "question"]
    )

    logger.info("Building RetrievalQA (RAG) chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_for_rag,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    # 3) Direct LLM Pipeline (No Retrieval)
    logger.info("Building direct LLM pipeline (no retrieval)...")
    direct_llm_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.0,
        device=device
    )

    def answer_with_rag(question: str, user_role: str = "employee") -> str:
        """
        Use the RAG pipeline (RetrievalQA) to answer a given question.
        Apply alignment checks before returning.
        """
        response = qa_chain.invoke(question)
        raw_answer = response["result"]
        aligned = align_answer(raw_answer, user_role=user_role)
        return aligned

    def answer_with_llm_direct(question: str, user_role: str = "employee") -> str:
        """
        Use the direct pipeline (no retrieval) to answer a question.
        Apply alignment checks before returning.
        """
        direct_prompt = f"""
You are an AI assistant specialized in enterprise knowledge. Answer the following question thoroughly.
If you are not sure, say "I don't have enough information."

Question: {question}

Answer:
"""
        raw_output = direct_llm_pipeline(direct_prompt.strip())
        raw_answer = raw_output[0]["generated_text"]
        aligned = align_answer(raw_answer, user_role=user_role)
        return aligned

    # 5) Compare Answers in Interactive Loop
    logger.info("Entering interactive question loop. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            query = input("\nEnter your question (or 'exit'/'quit' to stop): ")
        except EOFError:
            logger.info("EOF encountered. Exiting.")
            break

        if query.lower() in ["exit", "quit"]:
            logger.info("User requested exit.")
            break

        # (a) Direct LLM answer
        default_answer = answer_with_llm_direct(query)
        # (b) RAG-based answer
        rag_answer = answer_with_rag(query)

        # Print comparison
        print("\n=== COMPARISON ===")
        print(f"QUESTION: {query}")
        print("\n[DEFAULT LLM ANSWER (no retrieval)]:")
        print(default_answer)
        print("-" * 40)
        print("[RAG ANSWER (with retrieval)]:")
        print(rag_answer)
        print("===")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Exiting.")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        sys.exit(1)
