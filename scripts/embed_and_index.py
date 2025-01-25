import json
import uuid
import torch
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from utils import load_config, get_logger, get_device
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    CollectionStatus,
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
JSONL_PATH = "../data/cleaned_data/all_processed_data.jsonl"  # Path to your JSONL file


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

# -------------------------------------------------------------------
# Main Upload Logic
# -------------------------------------------------------------------
def upload_jsonl_chunks_to_qdrant(
    qdrant_client: QdrantClient,
    model: SentenceTransformer,
    jsonl_path: str,
    collection_name: str,
    batch_size: int = BATCH_SIZE
) -> None:
    """
    Read a JSONL file, embed the chunks, and upsert them to a Qdrant collection.

    Expects each line in the JSONL to be a JSON object containing:
        - file_name
        - extension
        - text
        - chunk_id
        - (optionally) a pre-generated unique_id or anything else you need.

    Args:
        qdrant_client (QdrantClient): Instance of QdrantClient.
        model (SentenceTransformer): The embedding model.
        jsonl_path (str): Path to the JSONL file.
        collection_name (str): Name of the Qdrant collection.
        batch_size (int): Number of chunks to accumulate before upserting to Qdrant.
    """
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

            # Create metadata
            meta = {
                "source_file": data.get("file_name"),
                "extension": data.get("extension"),
                "chunk_id": data.get("chunk_id"),
                "text": text,
                "page_content": text,
                # Add more fields as needed
            }

            # Either use existing UUID or generate a new one
            # Here we simply generate a random UUID each time
            point_id = str(uuid.uuid4())

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
    qdrant_client = QdrantClient(url=QDRANT_URL)

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
