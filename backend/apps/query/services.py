# backend/apps/query/services.py

import chromadb
import asyncio
import os
from pathlib import Path
from django.conf import settings

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# MCP Integration imports (optional)
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain.agents import create_react_agent, AgentExecutor
    from langchain.tools.retriever import create_retriever_tool
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field, create_model
    from typing import Any, Dict, Optional
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

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

# ── 7) MCP Integration Helper Functions ────────────────────────────────────
def create_pydantic_model_from_schema(name: str, schema: Dict[str, Any]) -> BaseModel:
    """
    Create a Pydantic model from a JSON schema dictionary.
    This fixes the issue where MCP tools use dict schemas instead of Pydantic models.
    """
    if not MCP_AVAILABLE:
        return None
    
    properties = schema.get('properties', {})
    required = schema.get('required', [])
    
    # Create field definitions for the Pydantic model
    field_definitions = {}
    
    for field_name, field_schema in properties.items():
        field_type = field_schema.get('type', 'string')
        field_default = field_schema.get('default', ...)
        field_title = field_schema.get('title', field_name)
        
        # Convert JSON schema types to Python types
        if field_type == 'string':
            python_type = str
        elif field_type == 'integer':
            python_type = int
        elif field_type == 'number':
            python_type = float
        elif field_type == 'boolean':
            python_type = bool
        else:
            python_type = Any
        
        # Make field optional if not required and has a default
        if field_name not in required and field_default != ...:
            python_type = Optional[python_type]
            field_definitions[field_name] = (python_type, Field(default=field_default, title=field_title))
        elif field_name not in required:
            python_type = Optional[python_type]
            field_definitions[field_name] = (python_type, Field(default=None, title=field_title))
        else:
            field_definitions[field_name] = (python_type, Field(title=field_title))
    
    # Create the Pydantic model dynamically
    return create_model(name, **field_definitions)

def fix_mcp_tool_schemas(tools):
    """
    Fix MCP tools to use Pydantic models instead of dict schemas.
    This resolves the 'String tool inputs are not allowed when using tools with JSON schema args_schema' error.
    """
    if not MCP_AVAILABLE:
        return tools
    
    fixed_tools = []
    
    for tool in tools:
        if hasattr(tool, 'args_schema') and isinstance(tool.args_schema, dict):
            # Create a Pydantic model from the JSON schema
            model_name = f"{tool.name.replace('_', ' ').title().replace(' ', '')}Args"
            pydantic_model = create_pydantic_model_from_schema(model_name, tool.args_schema)
            
            # Create a new StructuredTool with the same functionality but proper schema
            fixed_tool = StructuredTool(
                name=tool.name,
                func=tool.func,
                description=tool.description,
                args_schema=pydantic_model,
                return_direct=tool.return_direct,
                verbose=tool.verbose,
                callbacks=tool.callbacks,
                tags=tool.tags,
                metadata=tool.metadata,
                handle_tool_error=tool.handle_tool_error,
                handle_validation_error=tool.handle_validation_error
            )
            fixed_tools.append(fixed_tool)
        else:
            # Keep tools that already have proper schemas
            fixed_tools.append(tool)
    
    return fixed_tools

def get_mcp_tools():
    """Get MCP tools from the ReportMiner MCP server."""
    if not MCP_AVAILABLE:
        return []
    
    try:
        backend_dir = Path(__file__).parent.parent.parent  # Fixed: go up to /backend
        mcp_server_path = backend_dir / "mcp_server.py"
        
        if not mcp_server_path.exists():
            print(f"Warning: MCP server not found at {mcp_server_path}")
            return []
        
        client = MultiServerMCPClient({
            "reportminer": {
                "command": "python",
                "args": [str(mcp_server_path)],
                "transport": "stdio"  
            }
        })
        
        # Get tools synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            tools = loop.run_until_complete(client.get_tools())
            if tools:
                # Fix schema compatibility issues
                fixed_tools = fix_mcp_tool_schemas(tools)
                return fixed_tools
            return []
        finally:
            loop.close()
    except Exception as e:
        print(f"Warning: Could not load MCP tools: {e}")
        print(f"Attempted MCP server path: {mcp_server_path if 'mcp_server_path' in locals() else 'path not set'}")
        # Uncomment next line for detailed debugging if needed
        # import traceback; traceback.print_exc()
        return []

def run_query_with_mcp(question: str) -> dict:
    """
    Enhanced query function that uses MCP tools alongside RAG retrieval.
    """
    try:
        # Get MCP tools
        mcp_tools = get_mcp_tools()
        
        if not mcp_tools:
            # Fallback to standard RAG if no MCP tools available
            return run_query(question)
        
        # Create a retriever tool for the agent (using different name to avoid conflict)
        retriever_tool = create_retriever_tool(
            retriever,
            "search_vector_documents", 
            "Search through the ReportMiner document collection using vector similarity."
        )
        
        # Combine MCP tools with retriever tool
        all_tools = mcp_tools + [retriever_tool]
        
        # Use local ReAct prompt template (equivalent to hub.pull("hwchase17/react"))
        from langchain.prompts import PromptTemplate
        prompt = PromptTemplate.from_template(
            "Answer the following questions as best you can. You have access to the following tools:\n\n"
            "{tools}\n\n"
            "Use the following format:\n\n"
            "Question: the input question you must answer\n"
            "Thought: you should always think about what to do\n"
            "Action: the action to take, should be one of [{tool_names}]\n"
            "Action Input: the input to the action\n"
            "Observation: the result of the action\n"
            "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
            "Thought: I now know the final answer\n"
            "Final Answer: the final answer to the original input question\n\n"
            "Begin!\n\n"
            "Question: {input}\n"
            "Thought:{agent_scratchpad}"
        )
        
        # Create LLM for agent
        llm = ChatOpenAI(
            temperature=0,
            model=settings.CHAT_MODEL_NAME,
            api_key=settings.OPENAI_API_KEY
        )
        
        # Create agent
        agent = create_react_agent(llm, all_tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=all_tools, 
            verbose=False,
            max_iterations=3,
            handle_parsing_errors=True
        )
        
        # Execute agent
        result = agent_executor.invoke({"input": question})
        
        return {
            "answer": result.get("output", "No answer generated"),
            "sources": [],  # MCP tools handle their own sourcing
            "mcp_used": True
        }
        
    except Exception as e:
        print(f"MCP query failed, falling back to standard RAG: {e}")
        return run_query(question)

# ── 8) Run Query Interface ──────────────────────────────────────────────────
def run_query(question: str, use_mcp: bool = None) -> dict:
    """
    Run the RAG chain on `question` and return a dict with 'answer' and 'sources'.
    
    Args:
        question: The question to ask
        use_mcp: Whether to use MCP tools. If None, uses environment variable MCP_ENABLED
    """
    # Check if MCP should be used
    if use_mcp is None:
        use_mcp = getattr(settings, 'MCP_ENABLED', False)
    
    # Use MCP-enhanced query if requested and available
    if use_mcp and MCP_AVAILABLE:
        return run_query_with_mcp(question)
    
    # Standard RAG query (original functionality preserved)
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
        "sources": sources,
        "mcp_used": False
    }
