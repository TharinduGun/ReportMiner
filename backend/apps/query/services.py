# apps/query/services.py

# 1) imports
import chromadb
from chromadb import PersistentClient
from chromadb.config import Settings
from django.conf import settings

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 2) initialize the Chroma client (using your settings value, no fallback)
chroma_client = PersistentClient(
    path=settings.CHROMA_PERSIST_DIR,
    settings=Settings()
)

# 3) configure the rest exactly as before
embedding_function = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=settings.OPENAI_API_KEY
)

vectordb = Chroma(
    client=chroma_client,
    collection_name=settings.CHROMA_COLLECTION_NAME,
    embedding_function=embedding_function,
)

retriever = vectordb.as_retriever(search_kwargs={"k": 10})

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant. Use the following context to answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(
        temperature=0,
        model=settings.CHAT_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY
    ),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_PROMPT}
)

def run_query(question: str) -> dict:
    result = qa_chain.invoke({"query": question})
    sources = [
        {**doc.metadata, "text": doc.page_content}
        for doc in result["source_documents"]
    ]
    return {
        "answer": result["result"],
        "sources": sources
    }
