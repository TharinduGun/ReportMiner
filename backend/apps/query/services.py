# backend/apps/query/services.py

import chromadb
from django.conf import settings
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# ── 1) Build a persistent Chroma client (no deprecated API) ────────────────
chroma_client = chromadb.PersistentClient(
    path=getattr(settings, "CHROMA_PERSIST_DIR", "./chroma")
)

# ── 2) Wrap it in LangChain’s Chroma vectorstore ─────────────────────────
vectordb = Chroma(
    client=chroma_client,
    collection_name=getattr(settings, "CHROMA_COLLECTION_NAME", "reportminer"),
    embedding_function=OpenAIEmbeddings(),  # or from langchain_community.embeddings if you upgrade
)

# ── 3) Create a Top-K retriever ────────────────────────────────────────────
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# ── 4) Build the RAG chain ────────────────────────────────────────────────
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)

def run_query(question: str) -> dict:
    """
    Run the RAG chain on `question` and return a dict with 'answer' and 'sources'.
    """
    result = qa_chain({"query": question})
    sources = [
        {
            **doc.metadata,          # chunk_id, section, page, etc.
            "text": doc.page_content # the actual text of the chunk
        }
        for doc in result["source_documents"]
    ]
    return {
        "answer":  result["result"],
        "sources": sources
    }
