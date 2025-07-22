# apps/ingestion/services/vector_store.py

import uuid
import logging
from typing import List, Dict, Any

from django.conf import settings

#from chromadb.config import Settings
import chromadb
logger = logging.getLogger(__name__)

# ── Initialize ChromaDB client & collection ─────────────────────────────────────
from chromadb.config import Settings

chroma_client = chromadb.PersistentClient(
    path=settings.CHROMA_PERSIST_DIR,
    settings=Settings(),
) 

collection = chroma_client.get_or_create_collection(
    name=getattr(settings, "CHROMA_COLLECTION_NAME", "reportminer")
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
        ValueError: If lengths of chunks and embeddings differ, or if required fields missing.
        RuntimeError: On failure to add to ChromaDB.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunks length {len(chunks)} != embeddings length {len(embeddings)}"
        )

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    for chunk, vector in zip(chunks, embeddings):
        text = chunk.get("text")
        if text is None:
            raise ValueError("Each chunk must include a 'text' field")
        chunk_id = str(uuid.uuid4())

        meta = dict(chunk.get("metadata", {}))
        meta.update({
            "chunk_id":   chunk_id,
            "token_count": chunk.get("token_count"),
        })
        clean_meta = {k: v for k, v in meta.items() if v is not None}

        ids.append(chunk_id)
        docs.append(text)
        metas.append(clean_meta)

    try:
        collection.add(
            ids=ids,
            documents=docs,
            embeddings=embeddings,
            metadatas=metas,
        )
    except Exception as e:
        logger.error("ChromaDB vector insert failed: %s", e)
        raise RuntimeError("Failed to add vectors to ChromaDB") from e
