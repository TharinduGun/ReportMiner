import os
from typing import List
from langchain_openai import OpenAIEmbeddings

# Ensure your OPENAI_API_KEY is set in environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set")

# Initialize OpenAI Embeddings model with batching
embedding_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    chunk_size=100,
    openai_api_key=OPENAI_API_KEY
)

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using OpenAIEmbeddings.

    Args:
        texts: List of string chunks to embed.

    Returns:
        List of embedding vectors corresponding to each text.
    """
    if not texts:
        return []
    # Directly delegate to LangChain’s batcher
    return embedding_model.embed_documents(texts)
