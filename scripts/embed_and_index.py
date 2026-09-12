import hashlib
import json
import uuid
from pathlib import Path
import torch
from typing import Dict, List, Any, Optional
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from utils import load_config, get_logger, get_device, _PROJECT_ROOT
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    CollectionStatus,
    FieldCondition,
    Filter,
    MatchValue,
)

# 1) Load Configuration & Logger
# --------------------------------------------------
config = load_config()
logger = get_logger(__name__)

# Reading from YAML config
QDRANT_URL = config["qdrant"]["url"]
COLLECTION_NAME = config["qdrant"]["collection"]
LLM_MODEL_NAME = config["models"]["llm_model_name"]
EMBED_MODEL_NAME = config["models"]["embedding_model_name"]

logger.info("Starting embed_and_index.py")

BATCH_SIZE = 32  # Adjust batch size as needed
JSONL_PATH = _PROJECT_ROOT / "data" / "cleaned_data" / "all_processed_data.jsonl"


# -------------------------------------------------------------------
# Embedding Model
# -------------------------------------------------------------------
def load_embedding_model(model_name: str, device: torch.device) -> SentenceTransformer:
    """
    Load a SentenceTransformer embedding model onto the specified device.
    
    Args:
        model_name (str): Name or path of the embedding model.
        device (torch.device): Torch device to load the model on.
    
    Returns:
        SentenceTransformer: The loaded embedding model.
    """
    logger.info(f"Loading embedding model '{model_name}'...")
    return SentenceTransformer(model_name, device=device)

# -------------------------------------------------------------------
# Qdrant Client & Collection Setup
# -------------------------------------------------------------------
def create_collection_if_not_exists(
    qdrant_client: QdrantClient,
    collection_name: str,
    vector_size: int
) -> None:
    """
    Create a Qdrant collection if it doesn't exist.

    Args:
        qdrant_client (QdrantClient): An instance of QdrantClient.
        collection_name (str): Name of the Qdrant collection.
        vector_size (int): Dimension of the embedding vectors.
    """
    collections_response = qdrant_client.get_collections()
    existing_names = [c.name for c in collections_response.collections]

    if collection_name not in existing_names:
        logger.info(f"Collection '{collection_name}' not found. Creating a new collection...")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        logger.info(f"Collection '{collection_name}' created successfully.")
    else:
        logger.info(f"Collection '{collection_name}' already exists.")

# -------------------------------------------------------------------
# Text Embedding
# -------------------------------------------------------------------
def embed_texts(model: SentenceTransformer, texts: List[str]) -> List[List[float]]:
    """
    Embed a list of texts using the provided model.

    For E5-based models, it's common to prefix text with "passage:" or "query:".
    Here, we use the "passage:" prefix by default.

    Args:
        model (SentenceTransformer): The embedding model.
        texts (List[str]): List of raw text strings.

    Returns:
        List[List[float]]: A list of embedding vectors.
    """
    processed_texts = [f"passage: {t.strip()}" for t in texts]
    embeddings = model.encode(processed_texts, convert_to_numpy=True)
    return embeddings.tolist()

# -------------------------------------------------------------------
# Qdrant Upsert Functions
# -------------------------------------------------------------------
def flush_batch_to_qdrant(
    qdrant_client: QdrantClient,
    model: SentenceTransformer,
    texts: List[str],
    ids: List[str],
    metas: List[Dict[str, Any]],
    collection_name: str
) -> None:
    """
    Compute embeddings for a batch of texts and upsert them into Qdrant.

    Args:
        qdrant_client (QdrantClient): An instance of QdrantClient.
        model (SentenceTransformer): The embedding model.
        texts (List[str]): List of text strings to embed.
        ids (List[str]): List of unique point IDs.
        metas (List[Dict[str, Any]]): List of metadata dicts.
        collection_name (str): Name of the Qdrant collection.
    """
    vectors = embed_texts(model, texts)
    points = [
        {
            "id": ids[i],
            "vector": vectors[i],
            "payload": metas[i]
        }
        for i in range(len(vectors))
    ]

    qdrant_client.upsert(
        collection_name=collection_name,
        points=points
    )
    logger.debug(f"Upserted {len(points)} points into collection '{collection_name}'.")


def delete_document_points(
    qdrant_client: QdrantClient,
    collection_name: str,
    document_id: str,
) -> None:
    """Delete points for one document within one logical collection."""
    qdrant_client.delete(
        collection_name=collection_name,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        ),
    )


def _embedding_dimension(model: SentenceTransformer) -> int:
    """Return the model dimension without changing the configured model stack."""
    try:
        return model.get_sentence_embedding_dimension()
    except AttributeError:
        example_vector = model.encode(["passage: test"], convert_to_numpy=True)[0]
        return len(example_vector)


def index_chunk_records(
    qdrant_client: QdrantClient,
    model: SentenceTransformer,
    records: List[Dict[str, Any]],
    collection_name: str,
    replace_document_id: Optional[str] = None,
    batch_size: int = BATCH_SIZE,
) -> None:
    """Incrementally index one document's collection-aware chunk records.

    Records come from ``preprocess_document`` and already contain the source
    collection and SHA-256 document identity.  A changed document can request
    deletion of its formerly indexed SHA before deterministic upserts begin.
    """
    if not records:
        raise ValueError("Cannot index an empty chunk record list")

    document_ids = {record.get("document_id") for record in records}
    if len(document_ids) != 1 or not next(iter(document_ids)):
        raise ValueError("Chunk records must belong to exactly one document")
    document_id = next(iter(document_ids))

    create_collection_if_not_exists(
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        vector_size=_embedding_dimension(model),
    )

    if replace_document_id and replace_document_id != document_id:
        delete_document_points(qdrant_client, collection_name, replace_document_id)

    for batch_start in range(0, len(records), batch_size):
        batch = records[batch_start:batch_start + batch_size]
        texts = [record["text"].strip() for record in batch]
        ids = [_make_point_id(document_id, int(record["chunk_id"])) for record in batch]
        metadata = [
            {
                "collection_name": record["collection_name"],
                "document_id": document_id,
                "source_file": record["source_file"],
                "source_path": record["source_path"],
                "extension": record.get("extension"),
                "chunk_id": int(record["chunk_id"]),
                "text": record["text"],
                "page_content": record["text"],
            }
            for record in batch
        ]
        flush_batch_to_qdrant(
            qdrant_client,
            model,
            texts,
            ids,
            metadata,
            collection_name,
        )

# -------------------------------------------------------------------
# Main Upload Logic
# -------------------------------------------------------------------
def _build_raw_file_index(raw_dir: Path) -> Dict[str, Path]:
    """
    Walk raw_dir and build a mapping of filename → absolute path.
    Uses direct name comparison so that filenames containing glob
    special characters (e.g. '[2024]') are found correctly.
    """
    index: Dict[str, Path] = {}
    for p in raw_dir.iterdir():
        if p.is_dir():
            for child in p.iterdir():
                if child.is_file():
                    index[child.name] = child
        elif p.is_file():
            index[p.name] = p
    return index


def _document_id(file_path: Path, _cache: Dict[str, str] = {}) -> Optional[str]:
    """
    Return the SHA-256 hex digest of the source file bytes.
    Results are cached per resolved path so each file is read only once.
    Returns None if the file cannot be read.
    """
    key = str(file_path.resolve())
    if key not in _cache:
        try:
            _cache[key] = hashlib.sha256(file_path.read_bytes()).hexdigest()
        except OSError as e:
            logger.warning(f"Cannot read source file for document_id: {file_path}: {e}")
            _cache[key] = None
    return _cache[key]


def _make_point_id(document_id: str, chunk_id: int) -> str:
    """
    Derive a deterministic UUID5 point ID from document content hash + per-file chunk index.

    Same document content + same chunk_id → same point ID on every run.
    Different document content → different document_id → different point IDs.
    Requires that the JSONL contains exactly one record per (file_name, chunk_id) pair,
    which is guaranteed when preprocessing is run exactly once per machine.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}#{chunk_id}"))


def upload_jsonl_chunks_to_qdrant(
    qdrant_client: QdrantClient,
    model: SentenceTransformer,
    jsonl_path: str,
    collection_name: str,
    batch_size: int = BATCH_SIZE
) -> None:
    """
    Read a JSONL file, embed the chunks, and upsert them to a Qdrant collection.

    Point IDs are content-based (SHA-256 of source file bytes + chunk index),
    making re-runs idempotent: the same corpus always produces the same IDs,
    so Qdrant upsert updates rather than duplicates existing points.

    Expects each JSONL line to contain:
        - file_name   : original source filename (used to locate file under data/raw/)
        - extension   : file extension
        - text        : chunk text
        - chunk_id    : sequential chunk index within the source file

    Args:
        qdrant_client  : Instance of QdrantClient.
        model          : The embedding model.
        jsonl_path     : Path to the JSONL file.
        collection_name: Name of the Qdrant collection.
        batch_size     : Number of chunks to accumulate before upserting to Qdrant.
    """
    raw_dir = _PROJECT_ROOT / "data" / "raw"
    raw_index = _build_raw_file_index(raw_dir)
    logger.info(f"Raw file index built: {len(raw_index)} files found under {raw_dir}")

    batch_texts = []
    batch_ids = []
    batch_metadata = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_idx} skipped (JSON error): {e}")
                continue

            text = data.get("text", "").strip()
            if not text:
                logger.debug(f"Line {line_idx} has empty or missing 'text' field. Skipping.")
                continue

            file_name = data.get("file_name", "")
            chunk_id = data.get("chunk_id", 0)

            # --- Content-based stable point ID ---
            # document_id = sha256(source file bytes) — stable across machines/renames
            # point_id    = uuid5(NAMESPACE_URL, f"{document_id}#{chunk_id}")
            # chunk_id is the per-file sequential index produced by preprocessing
            # (enumerate(chunks, 0)), guaranteed unique within a file as long as
            # preprocessing is run exactly once per JSONL (enforced by the log file).
            src_path = raw_index.get(file_name)
            if src_path is not None:
                doc_id = _document_id(src_path)
            else:
                # Source file not on disk (e.g. previously deleted); fall back to
                # a hash of the filename string so the ID is still deterministic.
                logger.warning(
                    f"Line {line_idx}: source file '{file_name}' not found in "
                    f"{raw_dir}. Using filename-based fallback document ID."
                )
                doc_id = hashlib.sha256(file_name.encode()).hexdigest()

            point_id = _make_point_id(doc_id, chunk_id)

            # Create metadata (document_id added for future deletion/re-index)
            meta = {
                "source_file": file_name,
                "extension": data.get("extension"),
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "text": text,
                "page_content": text,
            }

            batch_texts.append(text)
            batch_ids.append(point_id)
            batch_metadata.append(meta)

            # If we reach the batch limit, flush the batch to Qdrant
            if len(batch_texts) >= batch_size:
                flush_batch_to_qdrant(
                    qdrant_client,
                    model,
                    batch_texts,
                    batch_ids,
                    batch_metadata,
                    collection_name
                )
                batch_texts.clear()
                batch_ids.clear()
                batch_metadata.clear()

    # Flush remaining items if they exist
    if batch_texts:
        flush_batch_to_qdrant(
            qdrant_client,
            model,
            batch_texts,
            batch_ids,
            batch_metadata,
            collection_name
        )

# -------------------------------------------------------------------
# Main Routine
# -------------------------------------------------------------------
def main():
    """
    Main entry point for loading embeddings, creating Qdrant collection,
    and uploading text chunks from a JSONL file.
    """
    # Determine device
    device = get_device()

    # Load model
    model = load_embedding_model(EMBED_MODEL_NAME, device=device)

    # Connect to Qdrant
    # check_compatibility=False suppresses the version mismatch warning between
    # qdrant-client 1.15.1 and the running server 1.19.1.  All APIs used here
    # (upsert, get_collections, get_collection, create_collection) are stable
    # across this version range and work correctly.
    qdrant_client = QdrantClient(url=QDRANT_URL, check_compatibility=False)

    # Get vector dimension
    # Some SentenceTransformer models expose get_sentence_embedding_dimension() 
    # to directly obtain the dimension, but not all do. If not, we can encode an example.
    try:
        vector_size = model.get_sentence_embedding_dimension()
    except AttributeError:
        example_vector = model.encode(["passage: test"], convert_to_numpy=True)[0]
        vector_size = len(example_vector)

    logger.info(f"Detected embedding vector size: {vector_size}")

    # Ensure collection is created
    create_collection_if_not_exists(qdrant_client, COLLECTION_NAME, vector_size)

    # Upload data from JSONL
    upload_jsonl_chunks_to_qdrant(
        qdrant_client,
        model,
        jsonl_path=JSONL_PATH,
        collection_name=COLLECTION_NAME,
        batch_size=BATCH_SIZE
    )

    # Verify collection status
    collection_info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
    logger.info(f"Collection status: {collection_info.status}")
    if collection_info.status == CollectionStatus.GREEN:
        logger.info("Collection is ready to serve queries!")
    else:
        logger.warning(f"Collection '{COLLECTION_NAME}' is not ready. Current status: {collection_info.status}")

if __name__ == "__main__":
    main()
