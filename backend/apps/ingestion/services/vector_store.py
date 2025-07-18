import uuid
from typing import List, Dict, Any
import chromadb
from django.conf import settings

# Initialize Chroma client with persistence (NEW API)
chroma_client = chromadb.PersistentClient(
    path=getattr(settings, 'CHROMA_PERSIST_DIR', './chroma')
)

# Create or get the collection for ingestion
collection = chroma_client.get_or_create_collection(
    name=getattr(settings, 'CHROMA_COLLECTION_NAME', 'reportminer')
)

def add_vectors(
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]]
) -> None:
    """
    Add a batch of embeddings to ChromaDB.

    Args:
        chunks: List of dicts with keys 'text', 'metadata', and 'token_count'.
        embeddings: List of embedding vectors matching order of chunks.

    Raises:
        ValueError: if lengths of chunks and embeddings differ.
        Exception: on ChromaDB add/persist failure.
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
        # Copy metadata and include token count and chunk_id
        meta = chunk.get('metadata', {}).copy()
        meta['token_count'] = chunk.get('token_count')
        meta['chunk_id'] = chunk_id
        metadatas.append(meta)

    try:
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        # Note: PersistentClient automatically persists, no need to call persist()
    except Exception as e:
        # Consider logging error here
        raise Exception(f"Failed to add vectors to ChromaDB: {e}")