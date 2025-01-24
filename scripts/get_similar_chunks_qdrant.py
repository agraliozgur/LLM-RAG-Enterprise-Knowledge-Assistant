import os
import json
import uuid
from typing import List
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, CollectionStatus
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_device() -> torch.device:
    """
    Determines the best available device (CUDA, MPS, or CPU) for model computation.

    Returns:
        torch.device: The selected device.
    """
    try:
        if torch.cuda.is_available():
            logging.info("Using CUDA for computation.")
            return torch.device('cuda')
    except Exception as e:
        logging.warning(f"CUDA availability check failed: {e}")

    try:
        if torch.backends.mps.is_available():
            logging.info("Using MPS for computation.")
            return torch.device('mps')
    except AttributeError:
        logging.warning("MPS support is not available.")
    except Exception as e:
        logging.warning(f"MPS availability check failed: {e}")

    logging.info("Falling back to CPU for computation.")
    return torch.device('cpu')

# Constants
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "enterprise_chunks"
TOP_K_RESULTS = 3

def load_embedding_model(model_name: str, device: torch.device) -> SentenceTransformer:
    """
    Loads the specified SentenceTransformer model on the given device.

    Args:
        model_name (str): The name of the embedding model.
        device (torch.device): The device to load the model on.

    Returns:
        SentenceTransformer: The loaded embedding model.
    """
    logging.info(f"Loading embedding model '{model_name}' on device '{device}'.")
    return SentenceTransformer(model_name, device=device)

def initialize_qdrant_client(url: str) -> QdrantClient:
    """
    Initializes the Qdrant client with the specified URL.

    Args:
        url (str): The URL of the Qdrant service.

    Returns:
        QdrantClient: The initialized Qdrant client.
    """
    logging.info(f"Connecting to Qdrant at '{url}'.")
    return QdrantClient(url=url)

def query_similar_chunks(
    user_query: str,
    model: SentenceTransformer,
    qdrant_client: QdrantClient,
    collection_name: str,
    top_k: int = TOP_K_RESULTS
) -> None:
    """
    Queries the Qdrant collection for chunks similar to the user query.

    Args:
        user_query (str): The user's input query.
        model (SentenceTransformer): The embedding model to encode the query.
        qdrant_client (QdrantClient): The Qdrant client for database operations.
        collection_name (str): The name of the Qdrant collection to search.
        top_k (int, optional): The number of top similar chunks to retrieve. Defaults to TOP_K_RESULTS.
    """
    # Encode the user query with a "query: " prefix to maintain consistency with stored embeddings
    logging.info("Encoding the user query.")
    query_embedding = model.encode([f"query: {user_query}"], convert_to_numpy=True)[0].tolist()

    # Perform the search in Qdrant
    logging.info(f"Searching for top {top_k} similar chunks in collection '{collection_name}'.")
    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k
    )

    # Display the search results
    logging.info("Search Results:")
    for hit in search_results:
        score = hit.score
        payload = hit.payload
        chunk_id = payload.get('chunk_id')
        source_file = payload.get('source_file')
        text = payload.get('text')
        logging.info(
            f"Score: {score:.4f}, "
            f"Chunk ID: {chunk_id}, "
            f"Source: {source_file}, "
            f"Text: {text}"
        )

def main() -> None:
    """
    The main function that orchestrates the loading of the model, initializing the Qdrant client,
    and querying similar chunks based on a user-provided query.
    """
    # Determine the computation device
    device = get_device()

    # Load the embedding model
    model = load_embedding_model(EMBEDDING_MODEL_NAME, device)

    # Initialize the Qdrant client
    qdrant_client = initialize_qdrant_client(QDRANT_URL)

    # Example user queries
    # Uncomment and modify the following lines to test different queries
    # user_query_str = "I tried speaking French, as carefully as I could..."
    # user_query_str = "In your synthetic EKM case studies, how does a mid-sized healthcare company’s Human Resources department measure improvements in knowledge retention?"
    user_query_str = "What knowledge management tools did the compliance department of a multinational finance corporation adopt, and what outcomes were observed?"

    # Query similar chunks
    query_similar_chunks(
        user_query=user_query_str,
        model=model,
        qdrant_client=qdrant_client,
        collection_name=COLLECTION_NAME,
        top_k=TOP_K_RESULTS
    )

if __name__ == "__main__":
    main()

"""
Sample Output:
2025-01-25 10:00:00 - INFO - Using CUDA for computation.
2025-01-25 10:00:00 - INFO - Loading embedding model 'sentence-transformers/all-MiniLM-L6-v2' on device 'cuda'.
2025-01-25 10:00:05 - INFO - Connecting to Qdrant at 'http://localhost:6333'.
2025-01-25 10:00:05 - INFO - Encoding the user query.
2025-01-25 10:00:06 - INFO - Searching for top 3 similar chunks in collection 'enterprise_chunks'.
2025-01-25 10:00:06 - INFO - Search Results:
2025-01-25 10:00:06 - INFO - Score: 0.9203, Chunk ID: 0, Source: task_management_automation.png, Text: ...
2025-01-25 10:00:06 - INFO - Score: 0.8587, Chunk ID: 7, Source: Twitter-Age Knowledge Management for You and Employees.pdf, Text: ...
2025-01-25 10:00:06 - INFO - Score: 0.8560, Chunk ID: 7, Source: How Infosys Built an Enterprise Knowledge Management Assistant Using Generative AI on AWS _ AWS Partner Network (APN) Blog.pdf, Text: ...
"""
