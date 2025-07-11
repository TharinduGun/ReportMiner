"""
Enhanced MCP Server for ReportMiner - FIXED VERSION
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
from rag_engine import get_rag_engine

# Create MCP server instance
server = Server("reportminer-enhanced")

# Global variables
rag_engine = None

def get_initialized_rag_engine():
    """Lazy initialization of RAG engine"""
    global rag_engine
    if rag_engine is None:
        rag_engine = get_rag_engine()
    return rag_engine

@server.list_tools()
async def list_tools() -> List[Tool]:
    """Define all 6 MCP tools for ReportMiner"""
    return [
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
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool execution with proper async/sync handling"""
    
    try:
        # Run sync functions in executor to avoid async context issues
        import asyncio
        loop = asyncio.get_event_loop()
        
        if name == "search_documents":
            return await loop.run_in_executor(None, handle_search_documents, arguments)
        elif name == "get_document_summary":
            return await loop.run_in_executor(None, handle_get_document_summary, arguments) 
        elif name == "list_recent_documents":
            return await loop.run_in_executor(None, handle_list_recent_documents, arguments)
        elif name == "query_natural_language":
            return await loop.run_in_executor(None, handle_query_natural_language, arguments)
        elif name == "extract_numerical_data":
            return await loop.run_in_executor(None, handle_extract_numerical_data, arguments)
        elif name == "create_chart":
            return await loop.run_in_executor(None, handle_create_chart, arguments)
        else:
            return [TextContent(type="text", text=f"❌ Unknown tool: {name}")]
            
    except Exception as e:
        error_msg = f"❌ Error in {name}: {str(e)}"
        print(error_msg)  # Log for debugging
        return [TextContent(type="text", text=error_msg)]

# === CORE DOCUMENT TOOL IMPLEMENTATIONS (NOW SYNC) ===

def handle_search_documents(args: Dict[str, Any]) -> List[TextContent]:
    """Search documents using existing functionality"""
    try:
        query = args.get("query", "")
        limit = args.get("limit", 5)
        
        # Use your existing Document model for search
        documents = Document.objects.filter(
            filename__icontains=query
        )[:limit]
        
        if not documents:
            # Try content search in text segments
            segments = DocumentTextSegment.objects.filter(
                content__icontains=query
            ).select_related('document')[:limit]
            
            unique_docs = {}
            for segment in segments:
                if segment.document.id not in unique_docs:
                    unique_docs[segment.document.id] = segment.document
                    
            documents = list(unique_docs.values())
        
        results = []
        for doc in documents:
            results.append(f"📄 {doc.filename} (ID: {doc.id})")
            results.append(f"   📁 Type: {doc.file_type} | Size: {doc.file_size or 'Unknown'} bytes")
            results.append(f"   📅 Uploaded: {doc.uploaded_at.strftime('%Y-%m-%d %H:%M')}")
            results.append(f"   🔄 Status: {doc.processing_status}")
            results.append("")
        
        if not results:
            return [TextContent(type="text", text=f"No documents found for query: '{query}'")]
            
        response = f"🔍 Found {len(documents)} documents for '{query}':\n\n" + "\n".join(results)
        return [TextContent(type="text", text=response)]
        
    except Exception as e:
        return [TextContent(type="text", text=f"❌ Search error: {str(e)}")]

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
    """Use RAG engine for natural language queries"""
    try:
        question = args.get("question", "")
        include_sources = args.get("include_sources", True)
        
        if not question:
            return [TextContent(type="text", text="❌ Question cannot be empty")]
        
        # Use your existing RAG engine
        rag = get_initialized_rag_engine()
        result = rag.query(question, include_sources=include_sources)
        
        if result['success']:
            response = f"🤖 AI Answer:\n{result['answer']}\n"
            
            if include_sources and result['sources']:
                response += f"\n📚 Sources ({len(result['sources'])}):\n"
                for i, source in enumerate(result['sources'][:3], 1):  # Limit to 3 sources
                    response += f"{i}. {source['filename']} (Segment {source['sequence_number']})\n"
                    response += f"   📄 {source['content'][:100]}...\n"
            
            response += f"\n📊 Query Info: {result['metadata']['sources_found']} sources found"
            return [TextContent(type="text", text=response)]
        else:
            return [TextContent(type="text", text=f"❌ Query failed: {result.get('error', 'Unknown error')}")]
            
    except Exception as e:
        return [TextContent(type="text", text=f"❌ AI query error: {str(e)}")]

# === MATHEMATICAL ANALYSIS TOOL ===

def handle_extract_numerical_data(args: Dict[str, Any]) -> List[TextContent]:
    """Extract numerical data from documents using your existing structured data"""
    try:
        document_id = args.get("document_id")
        data_types = args.get("data_types", ["numeric", "currency", "percentage"])
        
        # Build query based on document filter
        if document_id:
            key_values = DocumentKeyValue.objects.filter(document_id=document_id)
            structured_data = DocumentStructuredData.objects.filter(document_id=document_id)
        else:
            key_values = DocumentKeyValue.objects.all()
            structured_data = DocumentStructuredData.objects.all()
        
        results = []
        
        # Extract from key-value pairs
        if "numeric" in data_types:
            numeric_kvs = key_values.filter(value_numeric__isnull=False)
            if numeric_kvs.exists():
                results.append("💰 Numerical Values Found:")
                for kv in numeric_kvs[:10]:  # Limit results
                    results.append(f"   • {kv.key_name}: {kv.value_numeric}")
                results.append("")
        
        # Extract currency values (assuming they're in text with currency symbols)
        if "currency" in data_types:
            currency_kvs = key_values.filter(
                value_text__iregex=r'[\$€£¥]|USD|EUR|GBP'
            )
            if currency_kvs.exists():
                results.append("💵 Currency Values Found:")
                for kv in currency_kvs[:10]:
                    results.append(f"   • {kv.key_name}: {kv.value_text}")
                results.append("")
        
        # Extract from structured data (tables)
        numeric_cells = structured_data.filter(numeric_value__isnull=False)
        if numeric_cells.exists():
            results.append("📊 Table Data Found:")
            for cell in numeric_cells[:15]:  # Limit results
                results.append(f"   • {cell.column_name or 'Column'}: {cell.numeric_value}")
            results.append("")
        
        # Summary statistics
        total_numeric = key_values.filter(value_numeric__isnull=False).count()
        total_currency = currency_kvs.count() if "currency" in data_types else 0
        total_cells = numeric_cells.count()
        
        summary = f"""📈 Data Extraction Summary:
   • Numeric Key-Values: {total_numeric}
   • Currency Values: {total_currency}  
   • Table Numeric Cells: {total_cells}
   • Total Data Points: {total_numeric + total_currency + total_cells}
"""
        
        if not results:
            return [TextContent(type="text", text="📊 No numerical data found in the specified documents.")]
        
        final_response = "\n".join(results) + "\n" + summary
        return [TextContent(type="text", text=final_response)]
        
    except Exception as e:
        return [TextContent(type="text", text=f"❌ Data extraction error: {str(e)}")]

# === VISUALIZATION TOOL ===

def handle_create_chart(args: Dict[str, Any]) -> List[TextContent]:
    """Create charts from document data"""
    try:
        chart_type = args.get("chart_type", "bar")
        data_source = args.get("data_source", "")
        document_id = args.get("document_id")
        title = args.get("title", "Document Data Visualization")
        
        # Get numerical data for visualization
        if document_id:
            numeric_data = DocumentStructuredData.objects.filter(
                document_id=document_id,
                numeric_value__isnull=False
            )
        else:
            numeric_data = DocumentStructuredData.objects.filter(
                numeric_value__isnull=False
            )[:50]  # Limit for performance
        
        if not numeric_data.exists():
            return [TextContent(type="text", text="📊 No numerical data available for visualization.")]
        
        # Prepare data for plotting
        data_dict = {}
        for item in numeric_data:
            column_name = item.column_name or f"Column_{item.column_number}"
            if column_name not in data_dict:
                data_dict[column_name] = []
            data_dict[column_name].append(float(item.numeric_value))
        
        # Create chart based on type
        try:
            if chart_type == "bar":
                chart_result = create_bar_chart(data_dict, title)
            elif chart_type == "line":
                chart_result = create_line_chart(data_dict, title)
            elif chart_type == "pie":
                chart_result = create_pie_chart(data_dict, title)
            else:
                chart_result = create_bar_chart(data_dict, title)  # Default fallback
            
            return [TextContent(type="text", text=chart_result)]
            
        except Exception as chart_error:
            # Fallback to data summary if chart creation fails
            summary = f"📊 Chart Creation Failed - Data Summary:\n\n"
            for col, values in data_dict.items():
                summary += f"📈 {col}:\n"
                summary += f"   • Count: {len(values)}\n"
                summary += f"   • Average: {sum(values)/len(values):.2f}\n"
                summary += f"   • Min: {min(values):.2f}\n"
                summary += f"   • Max: {max(values):.2f}\n\n"
            
            summary += f"❌ Chart error: {str(chart_error)}"
            return [TextContent(type="text", text=summary)]
        
    except Exception as e:
        return [TextContent(type="text", text=f"❌ Visualization error: {str(e)}")]

def create_bar_chart(data_dict: Dict[str, List[float]], title: str) -> str:
    """Create a simple bar chart and return description"""
    try:
        # Calculate averages for each column
        averages = {col: sum(values)/len(values) for col, values in data_dict.items()}
        
        result = f"📊 Bar Chart: {title}\n\n"
        result += "📈 Column Averages:\n"
        
        # Sort by value for better presentation
        sorted_data = sorted(averages.items(), key=lambda x: x[1], reverse=True)
        
        for col, avg in sorted_data[:10]:  # Limit to top 10
            # Create simple ASCII bar
            bar_length = int(avg / max(averages.values()) * 20) if averages.values() else 0
            bar = "█" * bar_length
            result += f"   {col:<20} {bar} {avg:.2f}\n"
        
        result += f"\n📊 Total Columns: {len(data_dict)}"
        result += f"\n📈 Highest Average: {max(averages.values()):.2f}" if averages else ""
        
        return result
        
    except Exception as e:
        return f"❌ Bar chart error: {str(e)}"

def create_line_chart(data_dict: Dict[str, List[float]], title: str) -> str:
    """Create line chart description"""
    result = f"📈 Line Chart: {title}\n\n"
    
    for col, values in list(data_dict.items())[:5]:  # Limit to 5 columns
        if len(values) > 1:
            trend = "📈 Increasing" if values[-1] > values[0] else "📉 Decreasing"
            result += f"📊 {col}: {trend}\n"
            result += f"   Start: {values[0]:.2f} → End: {values[-1]:.2f}\n"
            result += f"   Change: {((values[-1] - values[0]) / values[0] * 100):.1f}%\n\n"
    
    return result

def create_pie_chart(data_dict: Dict[str, List[float]], title: str) -> str:
    """Create pie chart description"""
    totals = {col: sum(values) for col, values in data_dict.items()}
    grand_total = sum(totals.values())
    
    result = f"🥧 Pie Chart: {title}\n\n"
    result += "📊 Percentage Breakdown:\n"
    
    for col, total in sorted(totals.items(), key=lambda x: x[1], reverse=True)[:8]:
        percentage = (total / grand_total * 100) if grand_total > 0 else 0
        result += f"   • {col}: {percentage:.1f}% ({total:.2f})\n"
    
    return result

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
        
        print(f"🛠️ Available Tools: 6 (4 core + 2 enhanced)")
        print(f"📊 Ready for mathematical analysis and visualization!")
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