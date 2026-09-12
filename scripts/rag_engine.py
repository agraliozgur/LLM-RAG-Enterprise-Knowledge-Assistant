"""
rag_engine.py — Shared RAG engine for the Enterprise Knowledge Assistant.

Pipeline
--------
1. Embed the question with SentenceTransformer (no prefix — model is all-MiniLM-L6-v2).
2. Query Qdrant with `query_points()` to get top-k results WITH similarity scores.
3. Apply a configurable score threshold BEFORE calling the LLM.
   - If no chunk meets the threshold → return "not found" without calling the LLM.
   - If qualifying chunks exist     → build context from the top-3 and call flan-t5-large.
4. Truncate each chunk to MAX_CHUNK_TOKENS before building the prompt so the total
   input never silently overflows flan-t5-large's 512-token encoder limit.
5. Return a dict: { "answer", "sources", "above_threshold" }.

Default threshold
-----------------
Empirically calibrated on 10 known-answerable queries against the current corpus
(all-MiniLM-L6-v2, 1,235 chunks).  Observed score ranges:

  Known-answerable queries : 0.55 – 0.77  (mean 0.66)
  Clearly off-topic queries: 0.16 – 0.38  (mean 0.25)

A threshold of 0.45 sits safely above off-topic noise (max 0.38) while remaining
below the lowest known-answerable score (min 0.55), providing a clear separation
band of ~0.17.  This default can be overridden interactively via the Streamlit
slider (range 0.0 – 1.0, step 0.01).
"""

from __future__ import annotations

from typing import Any, Iterable

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from transformers import pipeline as hf_pipeline

from utils import load_config, get_logger, get_device

# ---------------------------------------------------------------------------
# Token-budget constants
# ---------------------------------------------------------------------------
# flan-t5-large has a 512-token encoder limit.
# We reserve ~50 tokens for the prompt template and question, leaving
# 462 tokens for context.  With top-3 chunks that is ~150 tokens per chunk.
# Any chunk whose tokenised length exceeds MAX_CHUNK_TOKENS is hard-truncated
# before the prompt is assembled to prevent silent overflow.
_MAX_CONTEXT_TOKENS: int = 450   # total budget for all context chunks combined
_MAX_CHUNK_TOKENS: int = 150     # per-chunk hard cap

# ---------------------------------------------------------------------------
# Module-level singletons (lazy-initialised)
# ---------------------------------------------------------------------------
_config: dict | None = None
_embed_model: SentenceTransformer | None = None
_qdrant_client: QdrantClient | None = None
_llm_pipeline: Any | None = None
_tokenizer: Any | None = None
_logger = get_logger(__name__)

NOT_FOUND_MSG = (
    "No sufficiently relevant information was found in the knowledge base "
    "for your question."
)

_PROMPT_TEMPLATE = (
    "You are an enterprise knowledge assistant. "
    "Answer the user's question using ONLY the context below. "
    "Do not fabricate information.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

_GENERAL_PROMPT_TEMPLATE = (
    "You are a helpful assistant. Answer the question using general knowledge. "
    "Clearly say when you are uncertain. This answer is NOT grounded in the "
    "user's knowledge bases.\n\nQuestion: {question}\n\nAnswer:"
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_config() -> dict:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        cfg = _get_config()
        model_name = cfg["models"]["embedding_model_name"]
        _logger.info(f"Loading embedding model '{model_name}'...")
        _embed_model = SentenceTransformer(model_name, device=get_device())
    return _embed_model


def _get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        cfg = _get_config()
        url = cfg["qdrant"]["url"]
        _logger.info(f"Connecting to Qdrant at {url}")
        _qdrant_client = QdrantClient(url=url, check_compatibility=False)
    return _qdrant_client


def _get_tokenizer() -> Any:
    """Return the LLM tokenizer (shared with the pipeline, loaded once)."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        cfg = _get_config()
        model_name = cfg["models"]["llm_model_name"]
        _logger.info(f"Loading tokenizer for '{model_name}'...")
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
    return _tokenizer


def _truncate_chunk(text: str, max_tokens: int = _MAX_CHUNK_TOKENS) -> str:
    """
    Truncate *text* to at most *max_tokens* LLM-tokenizer tokens.

    Decoding the truncated token IDs back to text preserves clean word
    boundaries instead of cutting mid-character.
    """
    tok = _get_tokenizer()
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tok.decode(ids[:max_tokens], skip_special_tokens=True)


def _get_llm_pipeline() -> Any:
    global _llm_pipeline
    if _llm_pipeline is None:
        cfg = _get_config()
        model_name = cfg["models"]["llm_model_name"]
        max_length = cfg.get("pipeline", {}).get("max_answer_length", 512)
        _logger.info(f"Loading LLM '{model_name}'...")
        from transformers import AutoModelForSeq2SeqLM
        tokenizer = _get_tokenizer()          # reuse singleton
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        _llm_pipeline = hf_pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=max_length,
            temperature=0.0,
            device=get_device(),
        )
    return _llm_pipeline


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Return at most ``max_tokens`` tokenizer tokens without splitting text."""
    if max_tokens <= 0:
        return ""
    tok = _get_tokenizer()
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tok.decode(ids[:max_tokens], skip_special_tokens=True)


def _build_prompt(question: str, sources: list[dict]) -> str:
    """Build a prompt that cannot exceed the FLAN-T5 encoder token budget."""
    tok = _get_tokenizer()
    model_limit = getattr(tok, "model_max_length", 512)
    if not isinstance(model_limit, int) or model_limit <= 0 or model_limit > 8192:
        model_limit = 512

    # Long questions must leave room for actual evidence.  The original question
    # is still used for retrieval; only the generation prompt is shortened.
    prompt_question = _truncate_to_tokens(question, max_tokens=100)
    context_parts: list[str] = []
    for source in sources[:3]:
        current_context = "\n\n---\n\n".join(context_parts)
        current_context_tokens = len(
            tok(current_context, add_special_tokens=False)["input_ids"]
        )
        remaining_context = _MAX_CONTEXT_TOKENS - current_context_tokens
        if remaining_context <= 0:
            break
        candidate = _truncate_to_tokens(
            _truncate_chunk(source["text"]), min(_MAX_CHUNK_TOKENS, remaining_context)
        )
        trial_parts = context_parts + [candidate]
        trial_context = "\n\n---\n\n".join(trial_parts)
        candidate_ids = tok(candidate, add_special_tokens=False)["input_ids"]
        while (
            len(tok(trial_context, add_special_tokens=False)["input_ids"])
            > _MAX_CONTEXT_TOKENS
            and candidate_ids
        ):
            candidate_ids = candidate_ids[:-1]
            candidate = tok.decode(candidate_ids, skip_special_tokens=True)
            trial_context = "\n\n---\n\n".join(context_parts + [candidate])
        if not candidate:
            break
        trial_prompt = _PROMPT_TEMPLATE.format(
            context=trial_context, question=prompt_question
        )
        token_count = len(tok(trial_prompt, add_special_tokens=True)["input_ids"])
        if token_count <= model_limit:
            context_parts.append(candidate)
            continue

        # Preserve ranked evidence by retaining as much of this candidate as
        # fits, while respecting the existing 450-token context ceiling.
        candidate_tokens = tok(candidate, add_special_tokens=False)["input_ids"]
        while candidate_tokens:
            candidate_tokens = candidate_tokens[:-1]
            shortened = tok.decode(candidate_tokens, skip_special_tokens=True)
            shortened_context = "\n\n---\n\n".join(context_parts + [shortened])
            shortened_prompt = _PROMPT_TEMPLATE.format(
                context=shortened_context, question=prompt_question
            )
            if len(tok(shortened_prompt, add_special_tokens=True)["input_ids"]) <= model_limit:
                if shortened:
                    context_parts.append(shortened)
                break

    return _PROMPT_TEMPLATE.format(
        context="\n\n---\n\n".join(context_parts), question=prompt_question
    )


def _retrieve_candidates(
    query_vector: list[float],
    collection_names: Iterable[str],
    score_threshold: float,
    top_k: int,
) -> list[Any]:
    """Query each target collection and globally rank comparable cosine scores."""
    client = _get_qdrant_client()
    candidates = []
    for collection_name in collection_names:
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
        ).points
        candidates.extend(results)
    return sorted(
        candidates,
        key=lambda result: (
            -result.score,
            result.payload.get("collection_name", ""),
            result.payload.get("document_id", ""),
            result.payload.get("chunk_id", 0),
        ),
    )


def answer_with_general_llm(question: str) -> str:
    """Generate an explicitly ungrounded answer on deliberate user request."""
    prompt = _GENERAL_PROMPT_TEMPLATE.format(
        question=_truncate_to_tokens(question, max_tokens=180)
    )
    raw = _get_llm_pipeline()(prompt)
    return raw[0]["generated_text"].strip()


def answer_with_rag(
    question: str,
    score_threshold: float | None = None,
    top_k: int = 5,
    collection_names: Iterable[str] | None = None,
) -> dict:
    """
    Answer a question using Retrieval-Augmented Generation.

    Parameters
    ----------
    question        : The user's question.
    score_threshold : Minimum cosine similarity score for a chunk to qualify.
                      Default 0.45 is empirically calibrated on 10 known-answerable
                      queries against the current corpus (all-MiniLM-L6-v2, 1,235 chunks):
                        Known-answerable scores: 0.55 – 0.77 (mean 0.66)
                        Off-topic scores:        0.16 – 0.38 (mean 0.25)
                      0.45 sits in the clear separation band (~0.17 wide) between the
                      highest off-topic score (0.38) and the lowest known score (0.55).
                      Valid range: 0.0 – 1.0. Overridable via the Streamlit slider.
    top_k           : Number of candidates to retrieve from Qdrant before
                      threshold filtering.  Top-3 qualifying chunks are sent
                      to the LLM after per-chunk token truncation.

    Returns
    -------
    {
        "answer"         : str   — LLM answer, or NOT_FOUND_MSG,
        "sources"        : list  — list of source dicts (empty if no results),
        "above_threshold": bool  — True if the LLM was called,
    }

    Each source dict:
        { "source_file", "chunk_id", "text", "score" }
    Note: "text" in sources is the ORIGINAL (untruncated) chunk text for display.
    The LLM only sees the truncated version.
    """
    cfg = _get_config()
    if score_threshold is None:
        score_threshold = cfg.get("pipeline", {}).get("relevance_threshold", 0.45)
    target_collections = list(collection_names or [cfg["qdrant"]["collection"]])
    if not target_collections:
        return {
            "answer": NOT_FOUND_MSG,
            "sources": [],
            "above_threshold": False,
        }

    # 1. Embed the query (no "passage:" prefix — all-MiniLM-L6-v2 does not need it)
    embed_model = _get_embed_model()
    query_vector = embed_model.encode([question], convert_to_numpy=True)[0].tolist()

    # 2. Retrieve per collection, then merge/rank candidates before one answer.
    results = _retrieve_candidates(
        query_vector=query_vector,
        collection_names=target_collections,
        score_threshold=score_threshold,
        top_k=top_k,
    )[:top_k]

    # 3. Gate: if no qualifying results, return early WITHOUT calling the LLM
    if not results:
        _logger.info(
            f"No results above threshold {score_threshold:.2f} for query: {question!r}"
        )
        return {
            "answer": NOT_FOUND_MSG,
            "sources": [],
            "above_threshold": False,
        }

    # 4. Build sources list from qualifying results (preserves original full text)
    sources = [
        {
            "source_file": r.payload.get("source_file", "unknown"),
            "source_collection": r.payload.get("collection_name", "legacy"),
            "chunk_id": r.payload.get("chunk_id"),
            "text": r.payload.get("text", ""),
            "score": r.score,
        }
        for r in results
    ]

    # 5. Build LLM context from ranked chunks within the encoder token budget.
    top_chunks = sources[:3]
    prompt = _build_prompt(question, top_chunks)
    llm = _get_llm_pipeline()
    _logger.info(
        f"Calling LLM with {len(top_chunks)} context chunk(s), "
        f"top score={results[0].score:.4f}, "
        f"context_chunks={len(top_chunks)}"
    )
    raw = llm(prompt)
    answer = raw[0]["generated_text"].strip()

    return {
        "answer": answer,
        "sources": sources,
        "above_threshold": True,
    }
