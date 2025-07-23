# backend/apps/query/services.py

import chromadb
from django.conf import settings

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tiktoken

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

# ── 4) Create retriever with diverse chunk recall ─────────────────────────────
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

# ── 7) Token counting and safety functions ──────────────────────────────────────
encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Count tokens in text using OpenAI's encoding"""
    return len(encoding.encode(text))

def diversify_results(docs, max_docs=5):
    """Ensure diverse results from different data sources"""
    if not docs:
        return docs
    
    # Group documents by source
    source_groups = {}
    for doc in docs:
        source = doc.metadata.get('source', 'unknown')
        if source not in source_groups:
            source_groups[source] = []
        source_groups[source].append(doc)
    
    # Select documents from different sources
    diverse_docs = []
    sources_used = []
    
    # First, get one document from each unique source
    for source, group in source_groups.items():
        if len(diverse_docs) < max_docs:
            diverse_docs.append(group[0])  # Take the most relevant from each source
            sources_used.append(source)
    
    # Fill remaining slots with best matches
    remaining_slots = max_docs - len(diverse_docs)
    if remaining_slots > 0:
        for doc in docs:
            if len(diverse_docs) >= max_docs:
                break
            if doc not in diverse_docs:
                diverse_docs.append(doc)
    
    return diverse_docs[:max_docs]

def truncate_context_safely(docs, max_tokens=20000):
    """Truncate retrieved context to stay within token limits"""
    total_tokens = 0
    safe_docs = []
    
    for doc in docs:
        doc_tokens = count_tokens(doc.page_content)
        
        if total_tokens + doc_tokens <= max_tokens:
            safe_docs.append(doc)
            total_tokens += doc_tokens
        else:
            # If single doc is too large, truncate it
            if not safe_docs and doc_tokens > max_tokens:
                # Truncate the document content
                truncated_content = doc.page_content[:max_tokens*3]  # Rough character estimate
                while count_tokens(truncated_content) > max_tokens:
                    truncated_content = truncated_content[:-1000]  # Remove chunks until safe
                
                # Create new document with truncated content
                from langchain.schema import Document
                truncated_doc = Document(
                    page_content=truncated_content + "...[truncated for length]",
                    metadata=doc.metadata
                )
                safe_docs.append(truncated_doc)
            break
    
    return safe_docs

# ── 8) Run Query Interface ──────────────────────────────────────────────────
def run_query(question: str) -> dict:
    """
    Run the RAG chain on `question` and return a dict with 'answer' and 'sources'.
    """
    # First retrieve documents
    retrieved_docs = retriever.invoke(question)
    
    # Apply source diversification to avoid bias
    diverse_docs = diversify_results(retrieved_docs, max_docs=6)
    
    # Apply safety truncation
    safe_docs = truncate_context_safely(diverse_docs, max_tokens=20000)
    
    # Create a custom retriever that returns our safe documents
    class SafeRetriever:
        def get_relevant_documents(self, query):
            return safe_docs
        
        def invoke(self, input_data, config=None):
            return safe_docs
    
    # Temporarily replace retriever and run QA
    original_retriever = qa_chain.retriever
    qa_chain.retriever = SafeRetriever()
    
    try:
        result = qa_chain.invoke({"query": question})
    finally:
        # Restore original retriever
        qa_chain.retriever = original_retriever
    
    sources = [
        {
            **doc.metadata,           # Includes chunk_id, page, etc.
            "text": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content  # Truncate source text for response
        }
        for doc in safe_docs
    ]
    return {
        "answer": result["result"],
        "sources": sources
    }
