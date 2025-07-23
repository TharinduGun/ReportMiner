# apps/ingestion/services/vector_store.py

import uuid
import logging
from typing import List, Dict, Any

from django.conf import settings
from langchain_chroma import Chroma
#from chromadb.config import Settings
import chromadb
logger = logging.getLogger(__name__)

def sanitize_metadata(metadata_dict):
    """Ensure all metadata keys/values are valid for ChromaDB."""
    sanitized = {}
    for key, value in metadata_dict.items():
        # Ensure the key is a string (convert if not)
        str_key = str(key) if key is not None else "unknown_key"

        # Convert value to allowed types
        if isinstance(value, (str, int, float, bool)) or value is None:
            sanitized[str_key] = value
        elif isinstance(value, list):
            sanitized[str_key] = ", ".join(str(item) for item in value)
        else:
            sanitized[str_key] = str(value)
    return sanitized

# ── Initialize ChromaDB using LangChain wrapper for consistency ─────────────────
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import chromadb

# Use same embedding configuration as query service
embedding_function = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=settings.OPENAI_API_KEY
)

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(
    path=settings.CHROMA_PERSIST_DIR,
    settings=Settings(),
)

# Use LangChain Chroma wrapper for consistent embedding handling
vectordb = Chroma(
    client=chroma_client,
    collection_name=getattr(settings, "CHROMA_COLLECTION_NAME", "reportminer"),
    embedding_function=embedding_function,
)


def add_vectors(
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]]
) -> None:
    """
    Add a batch of documents to ChromaDB using LangChain Chroma wrapper.

    Args:
        chunks: List of dicts with keys 'text', 'metadata', and 'token_count'.
        embeddings: Pre-computed embeddings (not used with LangChain Chroma).

    Raises:
        ValueError: If required fields missing.
        RuntimeError: On failure to add to ChromaDB.
    """
    if not chunks:
        return

    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    for chunk in chunks:
        text = chunk.get("text")
        if text is None:
            raise ValueError("Each chunk must include a 'text' field")
        
        chunk_id = str(uuid.uuid4())
        
        meta = dict(chunk.get("metadata", {}))
        meta.update({
            "chunk_id": chunk_id,
            "token_count": chunk.get("token_count"),
        })
        # Sanitize metadata to ensure ChromaDB compatibility
        clean_meta = sanitize_metadata(meta)
        # Remove None values after sanitization
        clean_meta = {k: v for k, v in clean_meta.items() if v is not None}

        texts.append(text)
        metadatas.append(clean_meta)
        ids.append(chunk_id)

    try:
        # Use LangChain Chroma interface - it handles embeddings automatically
        vectordb.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Successfully added {len(texts)} documents to ChromaDB")
    except Exception as e:
        logger.error("ChromaDB vector insert failed: %s", e)
        raise RuntimeError("Failed to add vectors to ChromaDB") from e
