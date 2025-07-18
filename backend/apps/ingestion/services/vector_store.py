import uuid
from typing import List, Dict, Any
import chromadb
from django.conf import settings

# Initialize ChromaDB persistent client
chroma_client = chromadb.PersistentClient(
    path=getattr(settings, 'CHROMA_PERSIST_DIR', './chroma')
)

# Create or get the vector collection
collection = chroma_client.get_or_create_collection(
    name=getattr(settings, 'CHROMA_COLLECTION_NAME', 'reportminer')
)

def add_vectors(
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]]
) -> None:
    """
    Add a batch of embeddings to ChromaDB, sanitizing metadata to remove None values.

    Args:
        chunks: List of dicts with keys 'text', 'metadata', and 'token_count'.
        embeddings: Corresponding list of vector embeddings.

    Raises:
        ValueError: If lengths of chunks and embeddings differ.
        Exception: On failure to add to ChromaDB.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunks length {len(chunks)} != embeddings length {len(embeddings)}"
        )

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for chunk, vector in zip(chunks, embeddings):
        # Generate unique ID per chunk
        chunk_id = str(uuid.uuid4())
        ids.append(chunk_id)
        documents.append(chunk['text'])

        # Copy and enrich metadata, then drop None values
        meta = chunk.get('metadata', {}).copy()
        meta['chunk_id'] = chunk_id
        meta['token_count'] = chunk.get('token_count')
        clean_meta = {k: v for k, v in meta.items() if v is not None}
        metadatas.append(clean_meta)

    try:
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
    except Exception as e:
        raise Exception(f"Failed to add vectors to ChromaDB: {e}")
