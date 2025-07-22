# backend/apps/query/services.py

import chromadb
from django.conf import settings

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ── 1) Initialize Chroma client with persistent path ────────────────────────
chroma_client = chromadb.PersistentClient(
    path=getattr(settings, "CHROMA_PERSIST_DIR", "./chroma")
)

# ── 2) Configure embedding function (must match ingestion) ──────────────────
embedding_function = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=settings.OPENAI_API_KEY
)

# ── 3) Connect to Chroma vector store ───────────────────────────────────────
vectordb = Chroma(
    client=chroma_client,
    collection_name=settings.CHROMA_COLLECTION_NAME,
    embedding_function=embedding_function,
)

# ── 4) Create retriever with top-k chunk recall ─────────────────────────────
retriever = vectordb.as_retriever(search_kwargs={"k": 10})

# ── 5) Custom prompt for better grounding ───────────────────────────────────
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant. Use the following context to answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
)

# ── 6) Build the Retrieval-Augmented Generation (RAG) QA chain ──────────────
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(
        temperature=0,
        model=settings.CHAT_MODEL_NAME,  # e.g. "gpt-4o" or "gpt-3.5-turbo"
        api_key=settings.OPENAI_API_KEY
    ),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_PROMPT}
)

# ── 7) Run Query Interface ──────────────────────────────────────────────────
def run_query(question: str) -> dict:
    """
    Run the RAG chain on `question` and return a dict with 'answer' and 'sources'.
    """
    result = qa_chain({"query": question})
    sources = [
        {
            **doc.metadata,           # Includes chunk_id, page, etc.
            "text": doc.page_content  # Actual source text
        }
        for doc in result["source_documents"]
    ]
    return {
        "answer": result["result"],
        "sources": sources
    }
