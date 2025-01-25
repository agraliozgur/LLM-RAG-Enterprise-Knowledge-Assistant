#!/usr/bin/env python3
# FILE: compare_llm_vs_rag.py
# --------------------------------------------------
# Compare a direct LLM approach with an RAG (Retrieval-Augmented Generation) 
# approach using a Qdrant vector store for enterprise knowledge retrieval.
# --------------------------------------------------

import logging
import sys

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from qdrant_client import QdrantClient

# Local imports
from utils import load_config, get_logger, get_device
from alignment import align_answer

try:
    # Use official or community-based classes as appropriate
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Qdrant
    from langchain.llms import HuggingFacePipeline
except ImportError:
    # Fallback if you have a custom / community-based fork
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Qdrant
    from langchain_huggingface import HuggingFacePipeline


def main() -> None:
    """
    1) Load configuration and models.
    2) Build an RAG chain with Qdrant as the vector store.
    3) Build a direct LLM pipeline for comparison.
    4) Prompt user for questions and compare answers from:
       (a) Plain LLM
       (b) RAG-based approach
    """
    # --------------------------------------------------
    # 1) Load Configuration & Logger
    # --------------------------------------------------
    config = load_config()
    logger = get_logger(__name__)

    # Reading from YAML config
    qdrant_url = config["qdrant"]["url"]
    collection_name = config["qdrant"]["collection"]
    llm_model_name = config["models"]["llm_model_name"]
    embed_model_name = config["models"]["embedding_model_name"]

    logger.info("Starting compare_llm_vs_rag.py")

    # --------------------------------------------------
    # 2) Connect to Qdrant & Setup Embeddings
    # --------------------------------------------------
    logger.info(f"Using embedding model: {embed_model_name}")
    embedding_model = HuggingFaceEmbeddings(model_name=embed_model_name)

    logger.info(f"Connecting to Qdrant at: {qdrant_url}")
    qdrant_client = QdrantClient(url=qdrant_url)

    logger.info(f"Initializing Qdrant vector store with collection '{collection_name}'")
    vectorstore = Qdrant(
        client=qdrant_client,
        collection_name=collection_name,
        embeddings=embedding_model,
    )

    # --------------------------------------------------
    # 3) Load/Initialize LLM
    # --------------------------------------------------
    logger.info(f"Loading LLM model '{llm_model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)

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

    # --------------------------------------------------
    # 4) Build RAG Chain (RetrievalQA)
    # --------------------------------------------------
    prompt_template_text = """
You are an advanced enterprise knowledge assistant with access to a vector database of documents.
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

Remember, your final output should be concise, using correct grammar. Always ensure the user knows whether 
the information is from the database or general knowledge.

----
Database context (if any):
{context}

User Query:
{question}

Write your best possible answer:
""".strip()

    prompt = PromptTemplate(
        template=prompt_template_text,
        input_variables=["context", "question"]
    )

    logger.info("Building RetrievalQA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_for_rag,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # top-k passages
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    # --------------------------------------------------
    # 5) Direct LLM Pipeline (No Retrieval)
    # --------------------------------------------------
    logger.info("Setting up direct LLM pipeline for comparison...")

    def answer_with_llm_direct(question: str) -> str:
        """
        Use the direct pipeline (no retrieval) to answer a question.

        Args:
            question (str): The user's query.

        Returns:
            str: The generated answer from the direct LLM approach.
        """
        direct_prompt = f"""
You are an AI assistant specialized in enterprise knowledge. 
Answer the following question thoroughly.
If you are not sure, say "I don't have enough information."

Question: {question}

Answer:
""".strip()

        output = hf_pipeline(direct_prompt)
        return output[0]["generated_text"]

    def answer_with_rag(question: str) -> str:
        """
        Use the RAG pipeline (RetrievalQA) to answer a question.

        Args:
            question (str): The user's query.

        Returns:
            str: The generated answer from the RAG approach.
        """
        response = qa_chain.run(question)
        return response

    # --------------------------------------------------
    # 6) Interactive Loop for Comparison
    # --------------------------------------------------
    logger.info("Entering interactive question loop. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            query = input("\nEnter your question (or 'exit'/'quit' to stop): ")
        except EOFError:
            logger.info("EOF encountered. Exiting compare_llm_vs_rag.py.")
            break

        if query.lower() in ["exit", "quit"]:
            logger.info("User requested exit.")
            break

        # (a) Direct LLM answer
        default_answer = answer_with_llm_direct(query)
        # (b) RAG-based answer (with alignment)
        rag_raw = answer_with_rag(query)
        # Example: user_role could be determined by your user management system
        rag_aligned = align_answer(rag_raw, user_role="manager")

        print("\n=== COMPARISON ===")
        print(f"QUESTION: {query}")
        print("\n[DEFAULT LLM ANSWER (no retrieval)]:")
        print(default_answer)
        print("-" * 40)
        print("[RAG ANSWER (with retrieval, pre-alignment)]:")
        print(rag_raw)
        print("-" * 40)
        print("[RAG ANSWER (with alignment)]:")
        print(rag_aligned)
        print("===")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Exiting.")
        sys.exit(0)
    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")
        sys.exit(1)
