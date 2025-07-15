"""
Enhanced MCP Server for ReportMiner - FIXED VERSION WITH ALL ORIGINAL TOOLS
Provides document analysis, mathematical calculations, and visualization tools
"""
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set Django settings BEFORE any Django imports
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'reportminer.settings')

# Import Django and setup BEFORE any other imports
import django
django.setup()

# Standard library imports
import asyncio
import json
import base64
from typing import List, Dict, Any, Optional
from decimal import Decimal
from datetime import datetime
import io

# MCP imports (after Django setup)
from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent
from mcp.server.stdio import stdio_server

# Data analysis imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Django imports (after setup)
from apps.ingestion.models import Document, DocumentTextSegment, DocumentStructuredData, DocumentKeyValue
from apps.query.rag_engine import get_rag_engine

# Create MCP server instance
server = Server("reportminer-enhanced")

# Global variables
rag_engine = None

def get_initialized_rag_engine():
    """Lazy initialization of RAG engine"""
    global rag_engine
    if rag_engine is None:
        try:
            print("🧠 Initializing RAG engine...")
            rag_engine = get_rag_engine()
            health = rag_engine.health_check()
            print(f"✅ RAG engine initialized: {health['overall_status']}")
        except Exception as e:
            print(f"❌ RAG engine initialization failed: {e}")
            rag_engine = None
            raise
    return rag_engine

@server.list_tools()
async def list_tools() -> List[Tool]:
    """Define all 10 MCP tools for ReportMiner - KEEPING ALL ORIGINAL TOOLS"""
    return[
        # === CORE DOCUMENT TOOLS ===
        Tool(
            name="search_documents",
            description="Search through uploaded documents using text queries",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results", "default": 5}
                },
                "required": ["query"]
            }
        ),
        
        Tool(
            name="get_document_summary", 
            description="Get detailed information about a specific document including processing status and content statistics",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "string", "description": "UUID of document"}
                },
                "required": ["document_id"]
            }
        ),
        
        Tool(
            name="list_recent_documents",
            description="List recently uploaded documents with processing status",
            inputSchema={
                "type": "object", 
                "properties": {
                    "limit": {"type": "integer", "description": "Number of documents", "default": 10}
                }
            }
        ),
        
        Tool(
            name="query_natural_language",
            description="Ask natural language questions about document content using AI",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Natural language question"},
                    "include_sources": {"type": "boolean", "description": "Include source references", "default": True}
                },
                "required": ["question"]
            }
        ),
        
        # === MATHEMATICAL ANALYSIS TOOLS ===
        Tool(
            name="extract_numerical_data",
            description="Extract all numerical data from documents including currencies, percentages, dates, and quantities",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "string", "description": "Document UUID (optional - searches all if not provided)"},
                    "data_types": {
                        "type": "array", 
                        "items": {"type": "string"},
                        "description": "Filter by data types: numeric, currency, percentage, date",
                        "default": ["numeric", "currency", "percentage"]
                    }
                }
            }
        ),
        
        Tool(
            name="create_chart",
            description="Generate charts and visualizations from document data",
            inputSchema={
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string", 
                        "enum": ["bar", "line", "pie", "scatter"],
                        "description": "Type of chart to create"
                    },
                    "data_source": {"type": "string", "description": "Description of what data to visualize"},
                    "document_id": {"type": "string", "description": "Specific document (optional)"},
                    "title": {"type": "string", "description": "Chart title", "default": "Document Data Visualization"}
                },
                "required": ["chart_type", "data_source"]
            }
        ),
        
        Tool(
            name="calculate_metrics",
            description="Perform statistical analysis and calculations on extracted numerical data across all domains",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "string", "description": "Document UUID (optional - analyzes all if not provided)"},
                    "analysis_type": {
                        "type": "string",
                        "enum": ["basic_stats", "growth_analysis", "correlation", "trend_detection"],
                        "description": "Type of statistical analysis to perform",
                        "default": "basic_stats"
                    },
                    "time_period": {"type": "string", "description": "Time period for analysis (monthly, quarterly, yearly)"},
                    "comparison_baseline": {"type": "string", "description": "Baseline for comparison analysis"}
                }
            }
        ),
        
        Tool(
            name="domain_analysis",
            description="Intelligently detect document domain and perform domain-specific analysis (scientific, medical, engineering, financial, legal, research)",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "string", "description": "Document UUID to analyze"},
                    "force_domain": {
                        "type": "string", 
                        "enum": ["auto", "scientific", "medical", "engineering", "financial", "legal", "research"],
                        "description": "Force specific domain analysis or use auto-detection",
                        "default": "auto"
                    },
                    "include_recommendations": {"type": "boolean", "description": "Include domain-specific recommendations", "default": True}
                },
                "required": ["document_id"]
            }
        ),
        
        Tool(
            name="visualize_patterns",
            description="Create advanced visualizations and pattern analysis adapted to document domain and data type",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "string", "description": "Document UUID (optional)"},
                    "visualization_type": {
                        "type": "string",
                        "enum": ["time_series", "distribution", "correlation", "comparison", "trend_analysis"],
                        "description": "Type of visualization to create"
                    },
                    "domain_context": {"type": "string", "description": "Domain context for appropriate visualization"},
                    "data_range": {"type": "string", "description": "Date/time range for analysis"},
                    "comparison_groups": {"type": "array", "items": {"type": "string"}, "description": "Groups to compare"}
                },
                "required": ["visualization_type"]
            }
        ),
        
        Tool(
            name="generate_insights",
            description="AI-powered intelligent analysis to generate insights, patterns, and recommendations across all domains",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "string", "description": "Document UUID (optional)"},
                    "analysis_focus": {
                        "type": "string",
                        "enum": ["performance", "trends", "anomalies", "predictions", "recommendations", "comprehensive"],
                        "description": "Focus area for insight generation",
                        "default": "comprehensive"
                    },
                    "context_data": {"type": "string", "description": "Additional context for analysis"},
                    "comparison_benchmark": {"type": "string", "description": "Benchmark for comparative insights"}
                }
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Enhanced tool execution with proper async/sync handling - FIXED VERSION"""
    try:
        print(f"🔧 MCP Tool called: {name}")
        
        # Get event loop for running sync code in thread pool
        loop = asyncio.get_event_loop()
        
        # Route to appropriate tool handler using thread pool for sync code
        if name == "search_documents":
            return await loop.run_in_executor(None, handle_search_documents, arguments)
        elif name == "get_document_summary":
            return await loop.run_in_executor(None, handle_get_document_summary, arguments) 
        elif name == "list_recent_documents":
            return await loop.run_in_executor(None, handle_list_recent_documents, arguments)
        elif name == "query_natural_language":
            # Run RAG code in thread pool - FIXED VERSION
            return await loop.run_in_executor(None, handle_query_natural_language, arguments)
        elif name == "extract_numerical_data":
            return await loop.run_in_executor(None, handle_extract_numerical_data, arguments)
        elif name == "create_chart":
            return await loop.run_in_executor(None, handle_create_chart, arguments)
        elif name == "calculate_metrics":
            return await loop.run_in_executor(None, handle_calculate_metrics, arguments)
        elif name == "domain_analysis":
            return await loop.run_in_executor(None, handle_domain_analysis, arguments)
        elif name == "visualize_patterns":
            return await loop.run_in_executor(None, handle_visualize_patterns, arguments)
        elif name == "generate_insights":
            return await loop.run_in_executor(None, handle_generate_insights, arguments)
        else:
            return [TextContent(type="text", text=f"❌ Unknown tool: {name}")]
            
    except Exception as e:
        error_msg = f"❌ Error in {name}: {str(e)}"
        print(error_msg)  # Log for debugging
        return [TextContent(type="text", text=error_msg)]

# === CORE DOCUMENT TOOL IMPLEMENTATIONS (ALL SYNC) ===

def handle_search_documents(args: Dict[str, Any]) -> List[TextContent]:
    """Search documents using vector similarity - FIXED VERSION"""
    try:
        query = args.get("query", "")
        limit = args.get("limit", 5)
        
        if not query:
            return [TextContent(type="text", text="❌ Search query cannot be empty")]
        
        print(f"🔍 Vector Search: {query}")  # Debug log
        
        # Use your existing LangChain wrapper for vector search (KEY FIX)
        rag = get_initialized_rag_engine()
        relevant_docs = rag.get_similar_documents(query, k=limit)
        
        if not relevant_docs:
            return [TextContent(type="text", text=f"🔍 No documents found for query: '{query}'\n\nTry different search terms or check if documents are properly processed.")]
        
        # Format results to show actual content
        results = [f"🔍 **Found {len(relevant_docs)} relevant documents for '{query}':**\n"]
        
        for i, doc in enumerate(relevant_docs, 1):
            content = doc.get('content', 'No content available')
            metadata = doc.get('metadata', {})
            filename = metadata.get('filename', 'Unknown Document')
            sequence = metadata.get('sequence_number', 'N/A')
            distance = doc.get('distance', 'N/A')
            
            results.append(f"\n**{i}. {filename}** (Segment {sequence})")
            results.append(f"   📄 Content: \"{content}\"")
            results.append(f"   📊 Relevance Score: {distance}")
        
        response = "\n".join(results)
        return [TextContent(type="text", text=response)]
        
    except Exception as e:
        error_msg = f"❌ Search error: {str(e)}"
        print(error_msg)  # Log for debugging
        return [TextContent(type="text", text=error_msg)]

def handle_get_document_summary(args: Dict[str, Any]) -> List[TextContent]:
    """Get detailed document information"""
    try:
        doc_id = args.get("document_id")
        
        document = Document.objects.get(id=doc_id)
        
        # Get statistics
        segments_count = DocumentTextSegment.objects.filter(document=document).count()
        embedded_count = DocumentTextSegment.objects.filter(
            document=document, 
            embedding__isnull=False
        ).count()
        
        summary = f"""📊 Document Summary: {document.filename}

📄 Basic Information:
   • File Type: {document.file_type}
   • File Size: {document.file_size or 'Unknown'} bytes
   • Uploaded: {document.uploaded_at.strftime('%Y-%m-%d %H:%M:%S')}
   • Processing Status: {document.processing_status}

📝 Content Analysis:
   • Text Segments: {segments_count}
   • Embedded Segments: {embedded_count}
   • Processing Completed: {document.processing_completed_at or 'In Progress'}

🏷️ Classification:
   • Document Type: {document.document_type or 'Not classified'}
   • Language: {document.language}
   • Page Count: {document.page_count or 'Unknown'}
"""
        
        return [TextContent(type="text", text=summary)]
        
    except Document.DoesNotExist:
        return [TextContent(type="text", text=f"❌ Document not found: {doc_id}")]
    except Exception as e:
        return [TextContent(type="text", text=f"❌ Summary error: {str(e)}")]

def handle_list_recent_documents(args: Dict[str, Any]) -> List[TextContent]:
    """List recent documents"""
    try:
        limit = args.get("limit", 10)
        
        documents = Document.objects.order_by('-uploaded_at')[:limit]
        
        if not documents:
            return [TextContent(type="text", text="📁 No documents found in the system.")]
        
        results = [f"📚 {len(documents)} Most Recent Documents:\n"]
        
        for i, doc in enumerate(documents, 1):
            status_emoji = "✅" if doc.processing_status == "completed" else "🔄" if doc.processing_status == "processing" else "⏳"
            
            results.append(f"{i}. {status_emoji} {doc.filename}")
            results.append(f"    📅 {doc.uploaded_at.strftime('%Y-%m-%d %H:%M')} | {doc.file_type.upper()} | {doc.processing_status}")
            results.append("")
        
        return [TextContent(type="text", text="\n".join(results))]
        
    except Exception as e:
        return [TextContent(type="text", text=f"❌ List error: {str(e)}")]

def handle_query_natural_language(args: Dict[str, Any]) -> List[TextContent]:
    """Use RAG engine for natural language queries - COMPLETE SYNC VERSION"""
    try:
        question = args.get("question", "")
        include_sources = args.get("include_sources", True)
        
        if not question:
            return [TextContent(type="text", text="❌ Question cannot be empty")]
        
        print(f"🔍 MCP RAG Query: {question}")
        
        # Force synchronous execution context
        import django
        from django.db import connection
        connection.ensure_connection()
        
        # Use your working RAG engine (SYNCHRONOUS)
        rag = get_initialized_rag_engine()
        result = rag.query(question, include_sources=include_sources)
        
        print(f"📊 MCP RAG Result Success: {result.get('success', False)}")
        
        if result.get('success', False):
            answer = result.get('answer', 'No answer generated')
            response = f"📋 **Direct Answer from Your Documents:**\n\n{answer}\n"
            
            if include_sources and result.get('sources'):
                sources = result.get('sources', [])
                response += f"\n📚 **Source Documents ({len(sources)}):**\n"
                
                for i, source in enumerate(sources[:3], 1):
                    filename = source.get('filename', 'Unknown Document')
                    sequence = source.get('sequence_number', 'N/A')
                    content_preview = source.get('content', '')[:100]
                    
                    response += f"\n{i}. **{filename}** (Segment {sequence})\n"
                    response += f"   📄 \"{content_preview}...\"\n"
            
            metadata = result.get('metadata', {})
            sources_found = metadata.get('sources_found', 0)
            response += f"\n🔍 **Search Results:** {sources_found} relevant sources found"
            
            return [TextContent(type="text", text=response)]
            
        else:
            error_msg = result.get('error', 'Unknown RAG error')
            return [TextContent(type="text", text=f"❌ Could not find answer in documents: {error_msg}")]
            
    except Exception as e:
        error_msg = f"❌ MCP RAG integration error: {str(e)}"
        print(error_msg)
        return [TextContent(type="text", text=error_msg)]

# === STUB IMPLEMENTATIONS FOR ADVANCED TOOLS ===
# These return placeholder responses to avoid errors

def handle_extract_numerical_data(args: Dict[str, Any]) -> List[TextContent]:
    """Placeholder for numerical data extraction"""
    return [TextContent(type="text", text="📊 Numerical data extraction tool is under development. Basic functionality available.")]

def handle_create_chart(args: Dict[str, Any]) -> List[TextContent]:
    """Placeholder for chart creation"""
    return [TextContent(type="text", text="📈 Chart creation tool is under development. Visualization features coming soon.")]

def handle_calculate_metrics(args: Dict[str, Any]) -> List[TextContent]:
    """Placeholder for metrics calculation"""
    return [TextContent(type="text", text="📊 Metrics calculation tool is under development. Statistical analysis features coming soon.")]

def handle_domain_analysis(args: Dict[str, Any]) -> List[TextContent]:
    """Placeholder for domain analysis"""
    return [TextContent(type="text", text="🔬 Domain analysis tool is under development. Intelligent domain detection coming soon.")]

def handle_visualize_patterns(args: Dict[str, Any]) -> List[TextContent]:
    """Placeholder for pattern visualization"""
    return [TextContent(type="text", text="📊 Pattern visualization tool is under development. Advanced analytics coming soon.")]

def handle_generate_insights(args: Dict[str, Any]) -> List[TextContent]:
    """Placeholder for insight generation"""
    return [TextContent(type="text", text="🤖 AI insight generation tool is under development. Smart analysis features coming soon.")]

# === SERVER STARTUP ===

def check_system_status():
    """Synchronous system check before starting async server"""
    try:
        print("🚀 Starting Enhanced ReportMiner MCP Server...")
        
        # Test database connection (synchronous)
        doc_count = Document.objects.count()
        segment_count = DocumentTextSegment.objects.count()
        embedded_count = DocumentTextSegment.objects.filter(embedding__isnull=False).count()
        
        print(f"✅ Database connected:")
        print(f"   📄 Documents: {doc_count}")
        print(f"   📝 Text Segments: {segment_count}")
        print(f"   🔮 Embedded Segments: {embedded_count}")
        
        # Test structured data availability
        kv_count = DocumentKeyValue.objects.count()
        structured_count = DocumentStructuredData.objects.count()
        print(f"   🔢 Key-Value Pairs: {kv_count}")
        print(f"   📊 Structured Data Points: {structured_count}")
        
        # Test RAG engine
        try:
            rag = get_initialized_rag_engine()
            health = rag.health_check()
            print(f"✅ RAG Engine: {health['overall_status']}")
        except Exception as rag_error:
            print(f"⚠️ RAG Engine: {str(rag_error)}")
        
        print(f"🛠️ Available Tools: 10 (4 working + 6 placeholders)")
        print(f"🎯 Core Features: Document search, RAG queries, summaries")
        print(f"📊 Advanced Features: Coming soon (placeholders active)")
        print(f"🎯 Server ready for connections...")
        
        return True
        
    except Exception as e:
        print(f"❌ System check failed: {e}")
        return False

async def main():
    """Fixed async main function with proper sync/async separation"""
    try:
        # Run synchronous system check first
        if not check_system_status():
            return
        
        # Start async MCP server
        async with stdio_server(server) as streams:
            await server.run(*streams)
            
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
