#!/usr/bin/env python3
"""
FastMCP Server for ReportMiner
Provides MCP tools for document querying and retrieval
"""

import os
import sys
import django
import asyncio
from pathlib import Path

# Add the Django project to the path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'reportminer.settings')
django.setup()

from mcp.server.fastmcp import FastMCP
from apps.query.services import run_query, vectordb

# Initialize the MCP server
mcp = FastMCP("ReportMiner")

@mcp.tool()
def query_documents(question: str) -> str:
    """
    Query the ReportMiner document database using RAG (Retrieval-Augmented Generation).
    
    Args:
        question: The question to ask about the documents
        
    Returns:
        A comprehensive answer based on the ingested documents
    """
    try:
        result = run_query(question)
        # Format response with sources for MCP
        answer = result["answer"]
        sources = result.get("sources", [])
        
        if sources:
            source_info = "\n\nSources:\n"
            for i, source in enumerate(sources[:3], 1):  # Limit to top 3 sources
                doc_name = source.get("document_name", "Unknown")
                page = source.get("page", "N/A")
                source_info += f"{i}. {doc_name} (Page {page})\n"
            answer += source_info
            
        return answer
    except Exception as e:
        return f"Error querying documents: {str(e)}"

@mcp.tool()
def search_documents(query: str, limit: int = 5) -> str:
    """
    Search for relevant document chunks without generating an answer.
    
    Args:
        query: Search query to find relevant documents
        limit: Maximum number of results to return (default: 5)
        
    Returns:
        List of relevant document chunks with metadata
    """
    try:
        # Use the retriever directly for search
        docs = vectordb.similarity_search(query, k=limit)
        
        if not docs:
            return "No relevant documents found."
        
        results = []
        for i, doc in enumerate(docs, 1):
            doc_name = doc.metadata.get("document_name", "Unknown")
            page = doc.metadata.get("page", "N/A")
            chunk_id = doc.metadata.get("chunk_id", "N/A")
            
            # Truncate content for readability
            content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            
            results.append(f"{i}. Document: {doc_name} (Page {page}, Chunk {chunk_id})\n   Content: {content}\n")
        
        return "\n".join(results)
    except Exception as e:
        return f"Error searching documents: {str(e)}"

@mcp.tool()
def get_collection_stats() -> str:
    """
    Get statistics about the document collection.
    
    Returns:
        Information about the number of documents and chunks in the collection
    """
    try:
        # Get collection info from ChromaDB
        collection = vectordb._collection
        count = collection.count()
        
        return f"Document Collection Statistics:\n- Total chunks: {count}\n- Collection: {collection.name}"
    except Exception as e:
        return f"Error getting collection stats: {str(e)}"

@mcp.tool() 
def list_available_documents() -> str:
    """
    List all available documents in the collection.
    
    Returns:
        List of document names and basic metadata
    """
    try:
        # Query for unique document names
        results = vectordb.similarity_search("", k=100)  # Get many results to find unique docs
        
        # Extract unique document names
        doc_names = set()
        for doc in results:
            doc_name = doc.metadata.get("document_name", "Unknown")
            if doc_name != "Unknown":
                doc_names.add(doc_name)
        
        if not doc_names:
            return "No documents found in collection."
        
        doc_list = "\n".join([f"- {name}" for name in sorted(doc_names)])
        return f"Available Documents ({len(doc_names)} total):\n{doc_list}"
    except Exception as e:
        return f"Error listing documents: {str(e)}"

if __name__ == "__main__":
    # Run the MCP server
    mcp.run(transport="stdio")