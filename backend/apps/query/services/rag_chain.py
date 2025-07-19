import chromadb
from django.conf import settings
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Prompt template that instructs the LLM to ground its answer in provided context
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful AI assistant. Use the following context to answer the question as accurately as possible.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer: "
    )
)

def get_retriever(top_k: int = 5):
    """
    Connects to the persistent ChromaDB collection and returns a retriever
    that fetches the top_k most relevant chunks for a query.
    """
    client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    vector_store = Chroma(
        client=client,
        collection_name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
    )
    return vector_store.as_retriever(search_kwargs={"k": top_k})


def build_qa_chain(
    top_k: int = 5,
    llm_model: str = "gpt-4o",
    chain_type: str = "stuff"
):
    """
    Builds a RetrievalQA chain with:
      - A ChatOpenAI LLM (e.g., GPT-4o)
      - A Chroma retriever for semantic search
      - Optional custom prompt when using 'stuff' chain_type

    Returns a callable `qa_chain({"query": question})` ->
    {"result": ..., "source_documents": [...]}
    """
    # Initialize the language model
    llm = ChatOpenAI(
        model_name=llm_model,
        openai_api_key=settings.OPENAI_API_KEY,
        temperature=0
    )

    # Get the retriever for top-k chunks
    retriever = get_retriever(top_k)

    # Build the RAG QA chain
    if chain_type == "stuff":
        # for 'stuff', we can pass a custom prompt
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_PROMPT}
        )
    else:
        # other chain types use default prompts
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True
        )

    return qa_chain