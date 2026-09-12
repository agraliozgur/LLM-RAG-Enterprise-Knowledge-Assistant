# --------------------------------------------------
# Compare a direct LLM approach with an RAG approach.
# RAG path delegates entirely to rag_engine.answer_with_rag().
# --------------------------------------------------

import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from alignment import align_answer
from utils import load_config, get_logger, get_device
import rag_engine

# 1) Load Configuration & Logger
# --------------------------------------------------
config = load_config()
logger = get_logger(__name__)

QDRANT_URL = config["qdrant"]["url"]
COLLECTION_NAME = config["qdrant"]["collection"]
LLM_MODEL_NAME = config["models"]["llm_model_name"]
EMBED_MODEL_NAME = config["models"]["embedding_model_name"]

logger.info("Starting compare_llm_vs_rag.py")

# Default score threshold for CLI usage (overridable at runtime)
DEFAULT_SCORE_THRESHOLD = 0.50


# --------------------------------------------------
# Main Routine
# --------------------------------------------------
def main() -> None:
    """
    1) Build a direct LLM pipeline (no retrieval) for comparison.
    2) RAG path uses rag_engine.answer_with_rag() — no separate retrieval logic here.
    3) Prompt user for questions and compare answers from:
        (a) Plain LLM
        (b) RAG-based approach (with threshold gate and alignment)
    """
    device = get_device()
    logger.info(f"Loading LLM model '{LLM_MODEL_NAME}' for direct path...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)
    direct_llm_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.0,
        device=device,
    )

    def answer_with_llm_direct(question: str) -> str:
        """Direct LLM answer without retrieval."""
        prompt = (
            "You are an AI assistant specialized in enterprise knowledge. "
            "Answer the following question thoroughly. "
            "If you are not sure, say \"I don't have enough information.\"\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        raw = direct_llm_pipeline(prompt)
        return raw[0]["generated_text"]

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

        # (b) RAG answer via shared engine (threshold gate applied inside)
        rag_result = rag_engine.answer_with_rag(
            question=query,
            score_threshold=DEFAULT_SCORE_THRESHOLD,
        )
        rag_raw = rag_result["answer"]
        rag_sources = rag_result["sources"]
        rag_above = rag_result["above_threshold"]

        rag_answer_aligned = align_answer(rag_raw, user_role="manager")

        # Print comparison
        print("\n=== COMPARISON ===\n")
        print(f"QUESTION: {query}")
        print("\n[DEFAULT LLM ANSWER (no retrieval)]:")
        print(default_answer)
        print("-" * 40)
        print(f"[RAG ANSWER (threshold={DEFAULT_SCORE_THRESHOLD}, above={rag_above})]:")
        print(rag_answer_aligned)
        if rag_above and rag_sources:
            print("\n  Sources retrieved:")
            for i, src in enumerate(rag_sources, 1):
                print(
                    f"  [{i}] {src['source_file']}  "
                    f"chunk={src['chunk_id']}  score={src['score']:.4f}"
                )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Exiting.")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        sys.exit(1)
