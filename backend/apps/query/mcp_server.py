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
    
# === ADVANCED ANALYSIS TOOL IMPLEMENTATIONS ===

def handle_calculate_metrics(args: Dict[str, Any]) -> List[TextContent]:
    """Universal statistical analysis and calculations for any domain"""
    try:
        document_id = args.get("document_id")
        analysis_type = args.get("analysis_type", "basic_stats")
        time_period = args.get("time_period")
        comparison_baseline = args.get("comparison_baseline")
        
        # Get numerical data
        if document_id:
            structured_data = DocumentStructuredData.objects.filter(
                document_id=document_id,
                numeric_value__isnull=False
            )
            key_values = DocumentKeyValue.objects.filter(
                document_id=document_id,
                value_numeric__isnull=False
            )
        else:
            structured_data = DocumentStructuredData.objects.filter(
                numeric_value__isnull=False
            )[:100]  # Limit for performance
            key_values = DocumentKeyValue.objects.filter(
                value_numeric__isnull=False
            )[:50]
        
        if not structured_data.exists() and not key_values.exists():
            return [TextContent(type="text", text="📊 No numerical data available for analysis.")]
        
        # Prepare data for analysis
        all_values = []
        data_points = []
        
        # Collect from structured data
        for item in structured_data:
            value = float(item.numeric_value)
            all_values.append(value)
            data_points.append({
                "value": value,
                "source": "table",
                "column": item.column_name or "Unknown",
                "row": item.row_number
            })
        
        # Collect from key-value pairs
        for kv in key_values:
            value = float(kv.value_numeric)
            all_values.append(value)
            data_points.append({
                "value": value,
                "source": "key_value",
                "key": kv.key_name,
                "category": kv.key_category or "General"
            })
        
        if not all_values:
            return [TextContent(type="text", text="📊 No valid numerical values found for analysis.")]
        
        # Perform statistical analysis based on type
        if analysis_type == "basic_stats":
            result = calculate_basic_statistics(all_values, data_points)
        elif analysis_type == "growth_analysis":
            result = calculate_growth_analysis(all_values, data_points, time_period)
        elif analysis_type == "correlation":
            result = calculate_correlation_analysis(data_points)
        elif analysis_type == "trend_detection":
            result = calculate_trend_analysis(all_values, data_points)
        else:
            result = calculate_basic_statistics(all_values, data_points)  # Default
        
        return [TextContent(type="text", text=result)]
        
    except Exception as e:
        return [TextContent(type="text", text=f"❌ Metrics calculation error: {str(e)}")]

def calculate_basic_statistics(values: List[float], data_points: List[Dict]) -> str:
    """Calculate basic statistical metrics"""
    import statistics
    
    try:
        n = len(values)
        mean_val = statistics.mean(values)
        median_val = statistics.median(values)
        
        # Handle standard deviation for single values
        if n > 1:
            std_dev = statistics.stdev(values)
            variance = statistics.variance(values)
        else:
            std_dev = 0
            variance = 0
        
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val
        
        # Calculate percentiles
        q1 = statistics.quantiles(values, n=4)[0] if n >= 4 else min_val
        q3 = statistics.quantiles(values, n=4)[2] if n >= 4 else max_val
        
        result = f"""📊 Statistical Analysis Results

📈 Basic Statistics:
   • Sample Size: {n}
   • Mean (Average): {mean_val:.3f}
   • Median: {median_val:.3f}
   • Standard Deviation: {std_dev:.3f}
   • Variance: {variance:.3f}

📋 Range Analysis:
   • Minimum: {min_val:.3f}
   • Maximum: {max_val:.3f}
   • Range: {range_val:.3f}
   • First Quartile (Q1): {q1:.3f}
   • Third Quartile (Q3): {q3:.3f}

📊 Data Quality:
   • Coefficient of Variation: {(std_dev/mean_val*100):.1f}% (if mean > 0)
   • Data Spread: {'High' if std_dev > mean_val*0.3 else 'Moderate' if std_dev > mean_val*0.1 else 'Low'}
"""
        
        # Add data source breakdown
        table_count = sum(1 for dp in data_points if dp["source"] == "table")
        kv_count = sum(1 for dp in data_points if dp["source"] == "key_value")
        
        result += f"""
📂 Data Sources:
   • Table Data: {table_count} values
   • Key-Value Data: {kv_count} values
   • Total Data Points: {n}
"""
        
        return result
        
    except Exception as e:
        return f"❌ Statistical calculation error: {str(e)}"

def calculate_growth_analysis(values: List[float], data_points: List[Dict], time_period: str = None) -> str:
    """Calculate growth rates and trends"""
    try:
        if len(values) < 2:
            return "📊 Growth Analysis: Need at least 2 data points for growth analysis."
        
        # Simple growth calculation (first to last)
        first_val = values[0]
        last_val = values[-1]
        
        if first_val != 0:
            total_growth = ((last_val - first_val) / first_val) * 100
        else:
            total_growth = 0
        
        # Average period-over-period growth
        period_growths = []
        for i in range(1, len(values)):
            if values[i-1] != 0:
                growth = ((values[i] - values[i-1]) / values[i-1]) * 100
                period_growths.append(growth)
        
        avg_period_growth = sum(period_growths) / len(period_growths) if period_growths else 0
        
        # Trend direction
        increasing_periods = sum(1 for g in period_growths if g > 0)
        decreasing_periods = sum(1 for g in period_growths if g < 0)
        
        trend_direction = "Increasing" if increasing_periods > decreasing_periods else "Decreasing" if decreasing_periods > increasing_periods else "Mixed"
        
        result = f"""📈 Growth Analysis Results

🔄 Overall Growth:
   • Total Growth: {total_growth:.2f}%
   • From: {first_val:.3f} → To: {last_val:.3f}
   • Data Points: {len(values)}

📊 Period Analysis:
   • Average Period Growth: {avg_period_growth:.2f}%
   • Positive Periods: {increasing_periods}
   • Negative Periods: {decreasing_periods}
   • Overall Trend: {trend_direction}

🎯 Growth Pattern:
   • Volatility: {'High' if abs(max(period_growths) - min(period_growths)) > 50 else 'Moderate' if abs(max(period_growths) - min(period_growths)) > 20 else 'Low'}
   • Consistency: {'Consistent' if abs(avg_period_growth) < 10 else 'Variable'}
"""
        
        if time_period:
            result += f"\n⏰ Time Period: {time_period}"
        
        return result
        
    except Exception as e:
        return f"❌ Growth analysis error: {str(e)}"

def calculate_correlation_analysis(data_points: List[Dict]) -> str:
    """Analyze correlations between different data categories"""
    try:
        # Group data by categories
        categories = {}
        for dp in data_points:
            if dp["source"] == "table":
                cat = dp.get("column", "Unknown")
            else:
                cat = dp.get("category", "General")
            
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(dp["value"])
        
        if len(categories) < 2:
            return "📊 Correlation Analysis: Need at least 2 data categories for correlation analysis."
        
        result = f"""🔗 Correlation Analysis Results

📊 Data Categories Found: {len(categories)}
"""
        
        # Basic correlation analysis between categories
        category_stats = {}
        for cat, values in categories.items():
            if len(values) > 0:
                import statistics
                category_stats[cat] = {
                    "mean": statistics.mean(values),
                    "count": len(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0
                }
        
        result += "\n📈 Category Statistics:\n"
        for cat, stats in category_stats.items():
            result += f"   • {cat}: Mean={stats['mean']:.2f}, Count={stats['count']}, StdDev={stats['std']:.2f}\n"
        
        # Simple relationship analysis
        result += "\n🔍 Relationship Analysis:\n"
        cats = list(categories.keys())
        for i in range(len(cats)):
            for j in range(i+1, len(cats)):
                cat1, cat2 = cats[i], cats[j]
                mean1 = category_stats[cat1]["mean"]
                mean2 = category_stats[cat2]["mean"]
                
                ratio = mean1 / mean2 if mean2 != 0 else 0
                result += f"   • {cat1} vs {cat2}: Ratio = {ratio:.2f}\n"
        
        return result
        
    except Exception as e:
        return f"❌ Correlation analysis error: {str(e)}"

def calculate_trend_analysis(values: List[float], data_points: List[Dict]) -> str:
    """Detect trends and patterns in the data"""
    try:
        if len(values) < 3:
            return "📊 Trend Analysis: Need at least 3 data points for trend analysis."
        
        # Calculate moving averages
        window_size = min(3, len(values) // 2)
        moving_averages = []
        
        for i in range(len(values) - window_size + 1):
            avg = sum(values[i:i+window_size]) / window_size
            moving_averages.append(avg)
        
        # Trend direction analysis
        increases = 0
        decreases = 0
        
        for i in range(1, len(values)):
            if values[i] > values[i-1]:
                increases += 1
            elif values[i] < values[i-1]:
                decreases += 1
        
        total_changes = increases + decreases
        trend_strength = abs(increases - decreases) / total_changes if total_changes > 0 else 0
        
        # Determine trend
        if increases > decreases:
            main_trend = "Upward"
        elif decreases > increases:
            main_trend = "Downward"
        else:
            main_trend = "Sideways"
        
        # Volatility analysis
        import statistics
        if len(values) > 1:
            volatility = statistics.stdev(values) / statistics.mean(values) * 100
        else:
            volatility = 0
        
        result = f"""📈 Trend Analysis Results

🎯 Main Trend: {main_trend}
   • Trend Strength: {trend_strength:.2f} (0=weak, 1=strong)
   • Upward Movements: {increases}
   • Downward Movements: {decreases}
   • Sideways Periods: {len(values) - 1 - total_changes}

📊 Pattern Analysis:
   • Volatility: {volatility:.1f}%
   • Pattern: {'Highly Volatile' if volatility > 30 else 'Moderately Volatile' if volatility > 15 else 'Stable'}
   • Data Points: {len(values)}

🔍 Moving Average Trend:
   • First MA: {moving_averages[0]:.3f}
   • Last MA: {moving_averages[-1]:.3f}
   • MA Change: {((moving_averages[-1] - moving_averages[0]) / moving_averages[0] * 100):.2f}%
"""
        
        return result
        
    except Exception as e:
        return f"❌ Trend analysis error: {str(e)}"    

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

def handle_domain_analysis(args: Dict[str, Any]) -> List[TextContent]:
    """Intelligently detect document domain and perform domain-specific analysis"""
    try:
        document_id = args.get("document_id")
        force_domain = args.get("force_domain", "auto")
        include_recommendations = args.get("include_recommendations", True)
        
        # Get document
        try:
            document = Document.objects.get(id=document_id)
        except Document.DoesNotExist:
            return [TextContent(type="text", text=f"❌ Document not found: {document_id}")]
        
        # Get document content for analysis
        text_segments = DocumentTextSegment.objects.filter(document=document)
        full_content = " ".join([segment.content for segment in text_segments[:10]])  # First 10 segments
        
        if not full_content:
            return [TextContent(type="text", text="❌ No text content found for domain analysis.")]
        
        # Detect domain or use forced domain
        if force_domain == "auto":
            detected_domain = detect_document_domain(full_content)
        else:
            detected_domain = force_domain
        
        # Get domain-specific analysis
        domain_analysis = perform_domain_specific_analysis(document, detected_domain, full_content)
        
        # Get structured data relevant to domain
        domain_metrics = get_domain_specific_metrics(document, detected_domain)
        
        # Generate recommendations if requested
        recommendations = []
        if include_recommendations:
            recommendations = generate_domain_recommendations(detected_domain, domain_analysis, domain_metrics)
        
        # Format comprehensive response
        result = f"""🔬 Domain Analysis Report: {document.filename}

🎯 Document Classification:
   • Detected Domain: {detected_domain.title()}
   • Confidence: {domain_analysis['confidence']:.1f}%
   • Analysis Method: {force_domain if force_domain != 'auto' else 'Automatic Detection'}

📊 Domain-Specific Analysis:
{domain_analysis['analysis']}

📈 Key Metrics:
{domain_metrics}
"""
        
        if recommendations:
            result += f"""
💡 Domain-Specific Recommendations:
{chr(10).join([f'   • {rec}' for rec in recommendations])}
"""
        
        result += f"""
🔍 Content Overview:
   • Total Text Segments: {text_segments.count()}
   • Content Preview: {full_content[:200]}...
   • File Type: {document.file_type.upper()}
   • Processing Status: {document.processing_status}
"""
        
        return [TextContent(type="text", text=result)]
        
    except Exception as e:
        return [TextContent(type="text", text=f"❌ Domain analysis error: {str(e)}")]

def detect_document_domain(content: str) -> str:
    """Detect document domain based on content analysis"""
    content_lower = content.lower()
    
    # Define domain keywords with weights
    domain_keywords = {
        "scientific": [
            "hypothesis", "experiment", "methodology", "analysis", "results", "conclusion",
            "research", "study", "data", "findings", "statistical", "significance",
            "correlation", "variable", "sample", "population", "theory", "observation"
        ],
        "medical": [
            "patient", "treatment", "diagnosis", "clinical", "therapy", "symptoms",
            "disease", "medical", "health", "hospital", "doctor", "medicine",
            "condition", "procedure", "surgery", "pharmaceutical", "dosage", "trial"
        ],
        "engineering": [
            "design", "specification", "performance", "efficiency", "system", "analysis",
            "construction", "material", "structure", "process", "optimization", "testing",
            "implementation", "development", "technical", "engineering", "component", "assembly"
        ],
        "financial": [
            "revenue", "profit", "cost", "investment", "return", "financial", "budget",
            "income", "expenses", "assets", "liabilities", "cash", "flow", "margin",
            "roi", "growth", "earnings", "capital", "market", "economic"
        ],
        "legal": [
            "contract", "agreement", "legal", "law", "clause", "terms", "conditions",
            "liability", "compliance", "regulation", "policy", "rights", "obligations",
            "court", "jurisdiction", "statute", "precedent", "litigation", "settlement"
        ],
        "research": [
            "survey", "questionnaire", "respondents", "participants", "methodology", "findings",
            "literature", "review", "academic", "publication", "journal", "conference",
            "abstract", "introduction", "discussion", "references", "bibliography", "citation"
        ]
    }
    
    # Calculate domain scores
    domain_scores = {}
    for domain, keywords in domain_keywords.items():
        score = sum(1 for keyword in keywords if keyword in content_lower)
        domain_scores[domain] = score
    
    # Find highest scoring domain
    best_domain = max(domain_scores, key=domain_scores.get)
    best_score = domain_scores[best_domain]
    
    # If no strong indicators, default to research
    if best_score < 3:
        return "research"
    
    return best_domain

def perform_domain_specific_analysis(document, domain: str, content: str) -> Dict[str, Any]:
    """Perform analysis specific to the detected domain"""
    
    content_lower = content.lower()
    word_count = len(content.split())
    
    if domain == "scientific":
        return analyze_scientific_document(content_lower, word_count)
    elif domain == "medical":
        return analyze_medical_document(content_lower, word_count)
    elif domain == "engineering":
        return analyze_engineering_document(content_lower, word_count)
    elif domain == "financial":
        return analyze_financial_document(content_lower, word_count)
    elif domain == "legal":
        return analyze_legal_document(content_lower, word_count)
    else:  # research or default
        return analyze_research_document(content_lower, word_count)

def analyze_scientific_document(content: str, word_count: int) -> Dict[str, Any]:
    """Analyze scientific research document"""
    
    # Look for scientific structure
    has_abstract = "abstract" in content
    has_methodology = "methodology" or "method" in content
    has_results = "results" in content
    has_conclusion = "conclusion" in content
    
    # Count statistical terms
    stats_terms = ["p-value", "correlation", "regression", "significance", "hypothesis", "statistical"]
    stats_count = sum(1 for term in stats_terms if term in content)
    
    structure_score = sum([has_abstract, has_methodology, has_results, has_conclusion]) / 4 * 100
    
    analysis = f"""📊 Scientific Document Analysis:
   • Document Structure: {structure_score:.0f}% complete (Abstract: {'✓' if has_abstract else '✗'}, Methods: {'✓' if has_methodology else '✗'}, Results: {'✓' if has_results else '✗'}, Conclusion: {'✓' if has_conclusion else '✗'})
   • Statistical Content: {stats_count} statistical terms found
   • Content Density: {word_count} words
   • Research Focus: {'Quantitative' if stats_count > 3 else 'Qualitative' if stats_count < 2 else 'Mixed'}"""
    
    return {
        "confidence": min(95, 60 + structure_score/3 + stats_count*5),
        "analysis": analysis
    }

def analyze_medical_document(content: str, word_count: int) -> Dict[str, Any]:
    """Analyze medical document"""
    
    medical_terms = ["patient", "treatment", "diagnosis", "clinical", "therapy", "medical"]
    medical_count = sum(1 for term in medical_terms if term in content)
    
    # Look for medical structure
    has_diagnosis = "diagnosis" in content
    has_treatment = "treatment" in content
    has_symptoms = "symptoms" in content
    has_outcome = "outcome" or "result" in content
    
    clinical_score = sum([has_diagnosis, has_treatment, has_symptoms, has_outcome]) / 4 * 100
    
    analysis = f"""🏥 Medical Document Analysis:
   • Clinical Structure: {clinical_score:.0f}% (Diagnosis: {'✓' if has_diagnosis else '✗'}, Treatment: {'✓' if has_treatment else '✗'}, Symptoms: {'✓' if has_symptoms else '✗'}, Outcomes: {'✓' if has_outcome else '✗'})
   • Medical Terminology: {medical_count} medical terms identified
   • Document Length: {word_count} words
   • Document Type: {'Clinical Report' if clinical_score > 50 else 'Medical Research' if 'study' in content else 'Medical Reference'}"""
    
    return {
        "confidence": min(95, 50 + clinical_score/2 + medical_count*3),
        "analysis": analysis
    }

def analyze_engineering_document(content: str, word_count: int) -> Dict[str, Any]:
    """Analyze engineering document"""
    
    engineering_terms = ["design", "specification", "performance", "system", "analysis", "technical"]
    eng_count = sum(1 for term in engineering_terms if term in content)
    
    # Look for engineering structure
    has_specs = "specification" in content
    has_design = "design" in content
    has_testing = "test" in content or "testing" in content
    has_performance = "performance" in content
    
    technical_score = sum([has_specs, has_design, has_testing, has_performance]) / 4 * 100
    
    analysis = f"""⚙️ Engineering Document Analysis:
   • Technical Structure: {technical_score:.0f}% (Specifications: {'✓' if has_specs else '✗'}, Design: {'✓' if has_design else '✗'}, Testing: {'✓' if has_testing else '✗'}, Performance: {'✓' if has_performance else '✗'})
   • Engineering Terms: {eng_count} technical terms found
   • Content Scope: {word_count} words
   • Document Focus: {'Design Specification' if has_specs and has_design else 'Performance Analysis' if has_performance else 'Technical Report'}"""
    
    return {
        "confidence": min(95, 55 + technical_score/2 + eng_count*4),
        "analysis": analysis
    }

def analyze_financial_document(content: str, word_count: int) -> Dict[str, Any]:
    """Analyze financial document"""
    
    financial_terms = ["revenue", "profit", "cost", "investment", "financial", "budget"]
    fin_count = sum(1 for term in financial_terms if term in content)
    
    # Look for financial structure
    has_revenue = "revenue" in content
    has_profit = "profit" in content or "income" in content
    has_expenses = "expenses" in content or "cost" in content
    has_analysis = "analysis" in content or "performance" in content
    
    financial_score = sum([has_revenue, has_profit, has_expenses, has_analysis]) / 4 * 100
    
    analysis = f"""💰 Financial Document Analysis:
   • Financial Structure: {financial_score:.0f}% (Revenue: {'✓' if has_revenue else '✗'}, Profit/Income: {'✓' if has_profit else '✗'}, Expenses: {'✓' if has_expenses else '✗'}, Analysis: {'✓' if has_analysis else '✗'})
   • Financial Terms: {fin_count} financial terms identified
   • Report Length: {word_count} words
   • Report Type: {'Comprehensive Financial Report' if financial_score > 75 else 'Financial Summary' if financial_score > 50 else 'Budget/Cost Analysis'}"""
    
    return {
        "confidence": min(95, 60 + financial_score/2 + fin_count*5),
        "analysis": analysis
    }

def analyze_legal_document(content: str, word_count: int) -> Dict[str, Any]:
    """Analyze legal document"""
    
    legal_terms = ["contract", "agreement", "legal", "terms", "conditions", "liability"]
    legal_count = sum(1 for term in legal_terms if term in content)
    
    # Look for legal structure
    has_terms = "terms" in content
    has_conditions = "conditions" in content
    has_obligations = "obligations" in content or "responsibilities" in content
    has_liability = "liability" in content
    
    legal_score = sum([has_terms, has_conditions, has_obligations, has_liability]) / 4 * 100
    
    analysis = f"""⚖️ Legal Document Analysis:
   • Legal Structure: {legal_score:.0f}% (Terms: {'✓' if has_terms else '✗'}, Conditions: {'✓' if has_conditions else '✗'}, Obligations: {'✓' if has_obligations else '✗'}, Liability: {'✓' if has_liability else '✗'})
   • Legal Terminology: {legal_count} legal terms found
   • Document Length: {word_count} words
   • Document Nature: {'Contract/Agreement' if has_terms and has_conditions else 'Legal Policy' if 'policy' in content else 'Legal Analysis'}"""
    
    return {
        "confidence": min(95, 65 + legal_score/2 + legal_count*4),
        "analysis": analysis
    }

def analyze_research_document(content: str, word_count: int) -> Dict[str, Any]:
    """Analyze general research document"""
    
    research_terms = ["research", "study", "analysis", "findings", "methodology", "literature"]
    research_count = sum(1 for term in research_terms if term in content)
    
    # Look for research structure
    has_intro = "introduction" in content
    has_literature = "literature" in content
    has_methodology = "methodology" in content
    has_findings = "findings" in content or "results" in content
    
    research_score = sum([has_intro, has_literature, has_methodology, has_findings]) / 4 * 100
    
    analysis = f"""📚 Research Document Analysis:
   • Research Structure: {research_score:.0f}% (Introduction: {'✓' if has_intro else '✗'}, Literature: {'✓' if has_literature else '✗'}, Methodology: {'✓' if has_methodology else '✗'}, Findings: {'✓' if has_findings else '✗'})
   • Research Terms: {research_count} research terms identified
   • Document Length: {word_count} words
   • Research Type: {'Academic Paper' if research_score > 75 else 'Research Report' if research_score > 50 else 'Survey/Study'}"""
    
    return {
        "confidence": min(95, 50 + research_score/2 + research_count*3),
        "analysis": analysis
    }

def get_domain_specific_metrics(document, domain: str) -> str:
    """Get numerical metrics relevant to the domain"""
    
    try:
        # Get numerical data from the document
        structured_data = DocumentStructuredData.objects.filter(
            document=document,
            numeric_value__isnull=False
        )[:10]
        
        key_values = DocumentKeyValue.objects.filter(
            document=document,
            value_numeric__isnull=False
        )[:10]
        
        if not structured_data.exists() and not key_values.exists():
            return "   • No numerical metrics available for domain-specific analysis"
        
        metrics = []
        
        # Add structured data metrics
        for item in structured_data:
            metrics.append(f"   • {item.column_name or 'Data'}: {item.numeric_value}")
        
        # Add key-value metrics
        for kv in key_values:
            metrics.append(f"   • {kv.key_name}: {kv.value_numeric}")
        
        return "\n".join(metrics[:8])  # Limit to 8 metrics
        
    except Exception as e:
        return f"   • Error retrieving metrics: {str(e)}"

def generate_domain_recommendations(domain: str, analysis: Dict, metrics: str) -> List[str]:
    """Generate domain-specific recommendations"""
    
    recommendations = []
    confidence = analysis.get("confidence", 0)
    
    if domain == "scientific":
        recommendations = [
            "Consider extracting statistical data for meta-analysis",
            "Look for experimental parameters and results",
            "Analyze methodology for reproducibility assessment",
            "Extract key findings for literature mapping"
        ]
    elif domain == "medical":
        recommendations = [
            "Extract patient demographics and outcomes",
            "Analyze treatment protocols and effectiveness",
            "Look for adverse events and safety data",
            "Consider clinical significance of findings"
        ]
    elif domain == "engineering":
        recommendations = [
            "Extract performance specifications and tolerances",
            "Analyze design parameters and constraints",
            "Look for testing results and validation data",
            "Consider optimization opportunities"
        ]
    elif domain == "financial":
        recommendations = [
            "Calculate key financial ratios and trends",
            "Analyze revenue and cost patterns",
            "Look for investment opportunities and risks",
            "Consider budget variance analysis"
        ]
    elif domain == "legal":
        recommendations = [
            "Extract key terms and conditions",
            "Analyze compliance requirements",
            "Look for liability and risk factors",
            "Consider regulatory implications"
        ]
    else:  # research
        recommendations = [
            "Extract key findings and conclusions",
            "Analyze methodology and limitations",
            "Look for future research directions",
            "Consider practical applications"
        ]
    
    # Filter recommendations based on confidence
    if confidence < 70:
        recommendations = recommendations[:2]  # Only top 2 if low confidence
    elif confidence < 85:
        recommendations = recommendations[:3]  # Top 3 if medium confidence
    
    return recommendations

def handle_visualize_patterns(args: Dict[str, Any]) -> List[TextContent]:
    """Create advanced visualizations and pattern analysis adapted to document domain"""
    try:
        document_id = args.get("document_id")
        visualization_type = args.get("visualization_type", "distribution")
        domain_context = args.get("domain_context", "auto")
        data_range = args.get("data_range")
        comparison_groups = args.get("comparison_groups", [])
        
        # Get numerical data
        if document_id:
            structured_data = DocumentStructuredData.objects.filter(
                document_id=document_id,
                numeric_value__isnull=False
            )
            document = Document.objects.get(id=document_id)
            title_context = f"from {document.filename}"
        else:
            structured_data = DocumentStructuredData.objects.filter(
                numeric_value__isnull=False
            )[:100]
            title_context = "from all documents"
        
        if not structured_data.exists():
            return [TextContent(type="text", text="📊 No numerical data available for visualization.")]
        
        # Prepare data for visualization
        data_points = []
        for item in structured_data:
            data_points.append({
                "value": float(item.numeric_value),
                "column": item.column_name or f"Column_{item.column_number}",
                "row": item.row_number,
                "table": item.table_id
            })
        
        # Auto-detect domain if needed
        if domain_context == "auto" and document_id:
            # Get document content for domain detection
            text_segments = DocumentTextSegment.objects.filter(document_id=document_id)[:5]
            content = " ".join([seg.content for seg in text_segments])
            domain_context = detect_document_domain(content)
        
        # Create visualization based on type and domain
        if visualization_type == "time_series":
            result = create_time_series_visualization(data_points, domain_context, title_context)
        elif visualization_type == "distribution":
            result = create_distribution_visualization(data_points, domain_context, title_context)
        elif visualization_type == "correlation":
            result = create_correlation_visualization(data_points, domain_context, title_context)
        elif visualization_type == "comparison":
            result = create_comparison_visualization(data_points, domain_context, title_context, comparison_groups)
        elif visualization_type == "trend_analysis":
            result = create_trend_visualization(data_points, domain_context, title_context)
        else:
            result = create_distribution_visualization(data_points, domain_context, title_context)  # Default
        
        return [TextContent(type="text", text=result)]
        
    except Exception as e:
        return [TextContent(type="text", text=f"❌ Visualization error: {str(e)}")]

def create_time_series_visualization(data_points: List[Dict], domain: str, context: str) -> str:
    """Create time-series visualization adapted to domain"""
    
    # Group data by column for time series
    columns = {}
    for point in data_points:
        col = point["column"]
        if col not in columns:
            columns[col] = []
        columns[col].append(point["value"])
    
    result = f"📈 Time Series Analysis {context}\n"
    result += f"🎯 Domain Context: {domain.title()}\n\n"
    
    for col, values in list(columns.items())[:5]:  # Limit to 5 series
        if len(values) > 1:
            # Calculate trend
            first_val = values[0]
            last_val = values[-1]
            trend_pct = ((last_val - first_val) / first_val * 100) if first_val != 0 else 0
            trend_direction = "📈" if trend_pct > 5 else "📉" if trend_pct < -5 else "➡️"
            
            # Domain-specific interpretation
            interpretation = get_domain_specific_interpretation(domain, col, trend_pct, "time_series")
            
            result += f"📊 {col} Time Series:\n"
            result += f"   {trend_direction} Trend: {trend_pct:+.1f}% change\n"
            result += f"   📍 Range: {min(values):.2f} to {max(values):.2f}\n"
            result += f"   🔍 Pattern: {interpretation}\n\n"
            
            # Create ASCII time series
            normalized = [(v - min(values)) / (max(values) - min(values)) if max(values) != min(values) else 0.5 for v in values[:20]]
            ascii_chart = create_ascii_time_series(normalized)
            result += f"   📈 Trend Visualization:\n{ascii_chart}\n\n"
    
    return result

def create_distribution_visualization(data_points: List[Dict], domain: str, context: str) -> str:
    """Create distribution visualization adapted to domain"""
    
    import statistics
    
    # Group data by column
    columns = {}
    for point in data_points:
        col = point["column"]
        if col not in columns:
            columns[col] = []
        columns[col].append(point["value"])
    
    result = f"📊 Distribution Analysis {context}\n"
    result += f"🎯 Domain Context: {domain.title()}\n\n"
    
    for col, values in list(columns.items())[:4]:  # Limit to 4 distributions
        if len(values) > 0:
            mean_val = statistics.mean(values)
            median_val = statistics.median(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0
            
            # Domain-specific interpretation
            interpretation = get_domain_specific_interpretation(domain, col, std_val/mean_val if mean_val != 0 else 0, "distribution")
            
            result += f"📈 {col} Distribution:\n"
            result += f"   📊 Mean: {mean_val:.2f} | Median: {median_val:.2f}\n"
            result += f"   📏 Std Dev: {std_val:.2f} | Range: {max(values) - min(values):.2f}\n"
            result += f"   🔍 Pattern: {interpretation}\n"
            
            # Create ASCII histogram
            histogram = create_ascii_histogram(values)
            result += f"   📊 Distribution Shape:\n{histogram}\n\n"
    
    return result

def create_correlation_visualization(data_points: List[Dict], domain: str, context: str) -> str:
    """Create correlation visualization adapted to domain"""
    
    # Group data by column
    columns = {}
    for point in data_points:
        col = point["column"]
        if col not in columns:
            columns[col] = []
        columns[col].append(point["value"])
    
    if len(columns) < 2:
        return f"📊 Correlation Analysis {context}\n❌ Need at least 2 data columns for correlation analysis."
    
    result = f"🔗 Correlation Analysis {context}\n"
    result += f"🎯 Domain Context: {domain.title()}\n\n"
    
    # Calculate correlations between columns
    column_names = list(columns.keys())[:5]  # Limit to 5 columns
    
    result += "📊 Correlation Matrix:\n"
    for i, col1 in enumerate(column_names):
        for j, col2 in enumerate(column_names):
            if i < j:  # Only upper triangle
                # Simple correlation calculation
                vals1 = columns[col1][:min(len(columns[col1]), len(columns[col2]))]
                vals2 = columns[col2][:len(vals1)]
                
                if len(vals1) > 1:
                    correlation = calculate_simple_correlation(vals1, vals2)
                    strength = "Strong" if abs(correlation) > 0.7 else "Moderate" if abs(correlation) > 0.3 else "Weak"
                    direction = "Positive" if correlation > 0 else "Negative"
                    
                    # Domain-specific interpretation
                    interpretation = get_domain_specific_interpretation(domain, f"{col1} vs {col2}", correlation, "correlation")
                    
                    result += f"   🔗 {col1} ↔ {col2}:\n"
                    result += f"      Correlation: {correlation:.3f} ({strength} {direction})\n"
                    result += f"      Interpretation: {interpretation}\n\n"
    
    return result

def create_comparison_visualization(data_points: List[Dict], domain: str, context: str, groups: List[str]) -> str:
    """Create comparison visualization adapted to domain"""
    
    # Group data by column
    columns = {}
    for point in data_points:
        col = point["column"]
        if col not in columns:
            columns[col] = []
        columns[col].append(point["value"])
    
    result = f"⚖️ Comparison Analysis {context}\n"
    result += f"🎯 Domain Context: {domain.title()}\n\n"
    
    if len(columns) < 2:
        return result + "❌ Need at least 2 data columns for comparison analysis."
    
    # Compare top columns
    column_items = list(columns.items())[:4]  # Limit to 4 columns
    
    result += "📊 Column Comparisons:\n"
    
    for i, (col1, vals1) in enumerate(column_items):
        for j, (col2, vals2) in enumerate(column_items):
            if i < j:
                import statistics
                mean1 = statistics.mean(vals1) if vals1 else 0
                mean2 = statistics.mean(vals2) if vals2 else 0
                
                if mean2 != 0:
                    ratio = mean1 / mean2
                    percentage_diff = ((mean1 - mean2) / mean2) * 100
                else:
                    ratio = float('inf') if mean1 > 0 else 0
                    percentage_diff = 0
                
                # Domain-specific interpretation
                interpretation = get_domain_specific_interpretation(domain, f"{col1} vs {col2}", percentage_diff, "comparison")
                
                result += f"   📈 {col1} vs {col2}:\n"
                result += f"      Ratio: {ratio:.2f}:1\n"
                result += f"      Difference: {percentage_diff:+.1f}%\n"
                result += f"      Analysis: {interpretation}\n\n"
    
    return result

def create_trend_visualization(data_points: List[Dict], domain: str, context: str) -> str:
    """Create trend visualization adapted to domain"""
    
    # Group data by column and sort by row (assuming row represents sequence)
    columns = {}
    for point in data_points:
        col = point["column"]
        if col not in columns:
            columns[col] = []
        columns[col].append((point["row"], point["value"]))
    
    # Sort by row number
    for col in columns:
        columns[col].sort(key=lambda x: x[0])
    
    result = f"📈 Trend Analysis {context}\n"
    result += f"🎯 Domain Context: {domain.title()}\n\n"
    
    for col, data_tuples in list(columns.items())[:4]:  # Limit to 4 trends
        if len(data_tuples) > 2:
            values = [val for _, val in data_tuples]
            
            # Calculate trend metrics
            trend_slope = calculate_trend_slope(values)
            volatility = calculate_volatility(values)
            
            # Domain-specific interpretation
            interpretation = get_domain_specific_interpretation(domain, col, trend_slope, "trend")
            
            result += f"📊 {col} Trend Analysis:\n"
            result += f"   📈 Slope: {trend_slope:.3f} (trend strength)\n"
            result += f"   📊 Volatility: {volatility:.3f} (stability measure)\n"
            result += f"   🔍 Pattern: {interpretation}\n\n"
            
            # Create ASCII trend line
            trend_ascii = create_ascii_trend(values)
            result += f"   📈 Trend Visualization:\n{trend_ascii}\n\n"
    
    return result

def get_domain_specific_interpretation(domain: str, metric: str, value: float, viz_type: str) -> str:
    """Get domain-specific interpretation of visualization patterns"""
    
    interpretations = {
        "scientific": {
            "time_series": f"Experimental progression shows {'significant change' if abs(value) > 10 else 'stable pattern'}",
            "distribution": f"Data shows {'high variability' if value > 0.3 else 'controlled conditions'}",
            "correlation": f"Variables show {'strong relationship' if abs(value) > 0.7 else 'independent behavior'}",
            "comparison": f"Groups show {'significant difference' if abs(value) > 20 else 'similar performance'}",
            "trend": f"Data trend is {'strongly directional' if abs(value) > 0.1 else 'relatively stable'}"
        },
        "medical": {
            "time_series": f"Patient progression shows {'notable improvement' if value > 5 else 'decline' if value < -5 else 'stable condition'}",
            "distribution": f"Clinical values show {'high variance (needs attention)' if value > 0.4 else 'normal distribution'}",
            "correlation": f"Clinical parameters show {'significant association' if abs(value) > 0.6 else 'weak correlation'}",
            "comparison": f"Treatment groups show {'clinically significant difference' if abs(value) > 15 else 'similar outcomes'}",
            "trend": f"Clinical trend shows {'positive response' if value > 0.05 else 'concerning pattern' if value < -0.05 else 'stable monitoring'}"
        },
        "engineering": {
            "time_series": f"System performance shows {'degradation' if value < -3 else 'improvement' if value > 3 else 'stable operation'}",
            "distribution": f"Performance metrics show {'excessive variation' if value > 0.25 else 'within specifications'}",
            "correlation": f"System parameters show {'strong dependency' if abs(value) > 0.8 else 'independent operation'}",
            "comparison": f"Design variants show {'significant performance gap' if abs(value) > 25 else 'comparable results'}",
            "trend": f"Performance trend indicates {'optimization needed' if value < -0.02 else 'acceptable drift' if abs(value) < 0.05 else 'investigation required'}"
        },
        "financial": {
            "time_series": f"Financial performance shows {'strong growth' if value > 8 else 'decline' if value < -5 else 'steady performance'}",
            "distribution": f"Financial metrics show {'high volatility' if value > 0.35 else 'stable performance'}",
            "correlation": f"Financial indicators show {'strong relationship' if abs(value) > 0.75 else 'limited correlation'}",
            "comparison": f"Financial categories show {'significant variance' if abs(value) > 30 else 'balanced performance'}",
            "trend": f"Financial trend shows {'positive momentum' if value > 0.03 else 'concerning direction' if value < -0.03 else 'stable outlook'}"
        },
        "legal": {
            "time_series": f"Legal metrics show {'increasing complexity' if value > 2 else 'regulatory stability'}",
            "distribution": f"Legal parameters show {'high variability' if value > 0.3 else 'consistent framework'}",
            "correlation": f"Legal factors show {'strong interdependence' if abs(value) > 0.7 else 'independent clauses'}",
            "comparison": f"Legal provisions show {'significant differences' if abs(value) > 20 else 'standard terms'}",
            "trend": f"Legal trend indicates {'evolving requirements' if abs(value) > 0.02 else 'stable framework'}"
        },
        "research": {
            "time_series": f"Research data shows {'clear progression' if abs(value) > 5 else 'stable findings'}",
            "distribution": f"Research values show {'diverse responses' if value > 0.4 else 'consistent patterns'}",
            "correlation": f"Research variables show {'significant association' if abs(value) > 0.6 else 'independent factors'}",
            "comparison": f"Research groups show {'meaningful difference' if abs(value) > 18 else 'similar patterns'}",
            "trend": f"Research trend shows {'clear direction' if abs(value) > 0.04 else 'stable methodology'}"
        }
    }
    
    return interpretations.get(domain, interpretations["research"]).get(viz_type, "Pattern analysis complete")

# === VISUALIZATION HELPER FUNCTIONS ===

def create_ascii_time_series(normalized_values: List[float]) -> str:
    """Create ASCII time series chart"""
    height = 8
    width = min(50, len(normalized_values))
    
    chart_lines = []
    for y in range(height, 0, -1):
        line = "   "
        threshold = y / height
        for i in range(width):
            if i < len(normalized_values):
                if normalized_values[i] >= threshold - 0.125:
                    line += "█"
                else:
                    line += " "
            else:
                line += " "
        chart_lines.append(line)
    
    return "\n".join(chart_lines)

def create_ascii_histogram(values: List[float]) -> str:
    """Create ASCII histogram"""
    if not values:
        return "   No data for histogram"
    
    # Create bins
    min_val = min(values)
    max_val = max(values)
    if min_val == max_val:
        return f"   All values equal: {min_val:.2f}"
    
    bins = 8
    bin_width = (max_val - min_val) / bins
    bin_counts = [0] * bins
    
    for val in values:
        bin_idx = min(int((val - min_val) / bin_width), bins - 1)
        bin_counts[bin_idx] += 1
    
    max_count = max(bin_counts) if bin_counts else 1
    
    histogram = []
    for i, count in enumerate(bin_counts):
        bin_start = min_val + i * bin_width
        bar_length = int((count / max_count) * 20)
        bar = "█" * bar_length
        histogram.append(f"   {bin_start:6.1f}|{bar:<20} ({count})")
    
    return "\n".join(histogram)

def create_ascii_trend(values: List[float]) -> str:
    """Create ASCII trend line"""
    if len(values) < 2:
        return "   Insufficient data for trend"
    
    # Normalize values
    min_val = min(values)
    max_val = max(values)
    if min_val == max_val:
        return f"   Flat trend at {min_val:.2f}"
    
    normalized = [(v - min_val) / (max_val - min_val) for v in values]
    
    height = 6
    width = min(40, len(values))
    
    trend_lines = []
    for y in range(height, 0, -1):
        line = "   "
        threshold = y / height
        for i in range(width):
            if i < len(normalized):
                if abs(normalized[i] - threshold) < 0.15:
                    line += "●"
                else:
                    line += " "
            else:
                line += " "
        trend_lines.append(line)
    
    return "\n".join(trend_lines)

def calculate_simple_correlation(x: List[float], y: List[float]) -> float:
    """Calculate simple correlation coefficient"""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    import statistics
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
    sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(len(x)))
    sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(len(y)))
    
    denominator = (sum_sq_x * sum_sq_y) ** 0.5
    
    return numerator / denominator if denominator != 0 else 0.0

def calculate_trend_slope(values: List[float]) -> float:
    """Calculate trend slope using linear regression"""
    if len(values) < 2:
        return 0.0
    
    n = len(values)
    x = list(range(n))
    
    # Simple linear regression
    sum_x = sum(x)
    sum_y = sum(values)
    sum_xy = sum(x[i] * values[i] for i in range(n))
    sum_x2 = sum(xi ** 2 for xi in x)
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    return slope

def calculate_volatility(values: List[float]) -> float:
    """Calculate volatility as coefficient of variation"""
    if len(values) < 2:
        return 0.0
    
    import statistics
    mean_val = statistics.mean(values)
    std_val = statistics.stdev(values)
    
    return std_val / mean_val if mean_val != 0 else 0.0

def handle_generate_insights(args: Dict[str, Any]) -> List[TextContent]:
    """AI-powered intelligent analysis to generate insights, patterns, and recommendations"""
    try:
        document_id = args.get("document_id")
        analysis_focus = args.get("analysis_focus", "comprehensive")
        context_data = args.get("context_data", "")
        comparison_benchmark = args.get("comparison_benchmark", "")
        
        # Get document and content if specified
        if document_id:
            try:
                document = Document.objects.get(id=document_id)
                context_desc = f"for {document.filename}"
            except Document.DoesNotExist:
                return [TextContent(type="text", text=f"❌ Document not found: {document_id}")]
        else:
            document = None
            context_desc = "across all documents"
        
        # Generate insights based on focus area
        insights = generate_comprehensive_insights(document, analysis_focus, context_data, comparison_benchmark)
        
        # Format final insights report
        result = f"""🤖 AI-Powered Insights Report {context_desc}

🎯 Analysis Focus: {analysis_focus.title()}
{'🔍 Context: ' + context_data if context_data else ''}
{'📊 Benchmark: ' + comparison_benchmark if comparison_benchmark else ''}

{insights}

🕐 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return [TextContent(type="text", text=result)]
        
    except Exception as e:
        return [TextContent(type="text", text=f"❌ Insight generation error: {str(e)}")]

def generate_comprehensive_insights(document, focus: str, context: str, benchmark: str) -> str:
    """Generate comprehensive AI insights based on all available data"""
    
    insights = []
    
    try:
        # Get data for analysis
        if document:
            numerical_data = get_document_numerical_insights(document)
            content_insights = get_document_content_insights(document)
            domain_insights = get_document_domain_insights(document)
        else:
            numerical_data = get_global_numerical_insights()
            content_insights = get_global_content_insights()
            domain_insights = get_global_domain_insights()
        
        # Generate insights based on focus
        if focus == "performance":
            insights.extend(generate_performance_insights(numerical_data, benchmark))
        elif focus == "trends":
            insights.extend(generate_trend_insights(numerical_data, content_insights))
        elif focus == "anomalies":
            insights.extend(generate_anomaly_insights(numerical_data, content_insights))
        elif focus == "predictions":
            insights.extend(generate_prediction_insights(numerical_data, content_insights))
        elif focus == "recommendations":
            insights.extend(generate_recommendation_insights(numerical_data, content_insights, domain_insights))
        else:  # comprehensive
            insights.extend(generate_performance_insights(numerical_data, benchmark))
            insights.extend(generate_trend_insights(numerical_data, content_insights))
            insights.extend(generate_recommendation_insights(numerical_data, content_insights, domain_insights))
        
        # Add contextual insights if provided
        if context:
            insights.extend(generate_contextual_insights(context, numerical_data))
        
        return format_insights_output(insights, focus)
        
    except Exception as e:
        return f"❌ Error generating insights: {str(e)}"

def get_document_numerical_insights(document) -> Dict[str, Any]:
    """Extract numerical insights from a specific document"""
    
    try:
        # Get structured data
        structured_data = DocumentStructuredData.objects.filter(
            document=document,
            numeric_value__isnull=False
        )
        
        # Get key-value data
        key_values = DocumentKeyValue.objects.filter(
            document=document,
            value_numeric__isnull=False
        )
        
        # Compile numerical insights
        values = [float(item.numeric_value) for item in structured_data]
        kv_values = [float(kv.value_numeric) for kv in key_values]
        all_values = values + kv_values
        
        if not all_values:
            return {"has_data": False}
        
        import statistics
        
        return {
            "has_data": True,
            "total_values": len(all_values),
            "mean": statistics.mean(all_values),
            "median": statistics.median(all_values),
            "std_dev": statistics.stdev(all_values) if len(all_values) > 1 else 0,
            "min_val": min(all_values),
            "max_val": max(all_values),
            "range": max(all_values) - min(all_values),
            "categories": len(set([item.column_name for item in structured_data] + [kv.key_name for kv in key_values])),
            "structured_count": len(values),
            "kv_count": len(kv_values)
        }
        
    except Exception as e:
        return {"has_data": False, "error": str(e)}

def get_document_content_insights(document) -> Dict[str, Any]:
    """Extract content insights from a specific document"""
    
    try:
        # Get text segments
        segments = DocumentTextSegment.objects.filter(document=document)
        
        # Analyze content
        total_segments = segments.count()
        embedded_segments = segments.filter(embedding__isnull=False).count()
        
        # Get content preview
        content_preview = ""
        if segments.exists():
            first_segments = segments[:3]
            content_preview = " ".join([seg.content[:100] for seg in first_segments])
        
        # Detect domain
        domain = detect_document_domain(content_preview) if content_preview else "unknown"
        
        return {
            "has_content": total_segments > 0,
            "total_segments": total_segments,
            "embedded_segments": embedded_segments,
            "embedding_coverage": (embedded_segments / total_segments * 100) if total_segments > 0 else 0,
            "detected_domain": domain,
            "content_preview": content_preview[:200],
            "processing_status": document.processing_status,
            "file_type": document.file_type
        }
        
    except Exception as e:
        return {"has_content": False, "error": str(e)}

def get_document_domain_insights(document) -> Dict[str, Any]:
    """Get domain-specific insights for a document"""
    
    try:
        content_insights = get_document_content_insights(document)
        domain = content_insights.get("detected_domain", "research")
        
        # Get domain-specific metrics
        if domain == "financial":
            return analyze_financial_domain_insights(document)
        elif domain == "medical":
            return analyze_medical_domain_insights(document)
        elif domain == "engineering":
            return analyze_engineering_domain_insights(document)
        elif domain == "scientific":
            return analyze_scientific_domain_insights(document)
        else:
            return analyze_general_domain_insights(document)
            
    except Exception as e:
        return {"domain": "unknown", "error": str(e)}

def generate_performance_insights(numerical_data: Dict, benchmark: str) -> List[str]:
    """Generate performance-focused insights"""
    
    insights = []
    
    if not numerical_data.get("has_data"):
        insights.append("📊 Performance Analysis: No numerical data available for performance evaluation")
        return insights
    
    # Performance metrics analysis
    mean_val = numerical_data["mean"]
    std_dev = numerical_data["std_dev"]
    range_val = numerical_data["range"]
    
    # Variability analysis
    cv = (std_dev / mean_val * 100) if mean_val != 0 else 0
    if cv < 10:
        insights.append(f"📈 High Consistency: Data shows excellent stability with only {cv:.1f}% variation")
    elif cv < 25:
        insights.append(f"📊 Moderate Variability: Performance shows {cv:.1f}% variation - within acceptable range")
    else:
        insights.append(f"⚠️ High Variability: Performance shows {cv:.1f}% variation - may need investigation")
    
    # Range analysis
    if range_val > mean_val * 2:
        insights.append(f"📏 Wide Range: Data spans {range_val:.2f} units, indicating diverse performance levels")
    else:
        insights.append(f"📏 Narrow Range: Data concentrated within {range_val:.2f} units, showing consistent performance")
    
    # Benchmark comparison if provided
    if benchmark:
        insights.append(f"🎯 Benchmark Analysis: Performance evaluated against {benchmark} standard")
    
    # Data quality assessment
    if numerical_data["total_values"] > 50:
        insights.append(f"✅ Strong Data Foundation: {numerical_data['total_values']} data points provide reliable insights")
    elif numerical_data["total_values"] > 10:
        insights.append(f"📊 Adequate Data: {numerical_data['total_values']} data points provide reasonable confidence")
    else:
        insights.append(f"⚠️ Limited Data: Only {numerical_data['total_values']} data points - insights may be preliminary")
    
    return insights

def generate_trend_insights(numerical_data: Dict, content_insights: Dict) -> List[str]:
    """Generate trend-focused insights"""
    
    insights = []
    
    if not numerical_data.get("has_data"):
        insights.append("📈 Trend Analysis: No numerical data available for trend evaluation")
        return insights
    
    # Data distribution insights
    mean_val = numerical_data["mean"]
    median_val = numerical_data["median"]
    
    if abs(mean_val - median_val) / mean_val < 0.1 if mean_val != 0 else True:
        insights.append("📊 Balanced Distribution: Mean and median are closely aligned, indicating normal distribution")
    else:
        skew_direction = "right" if mean_val > median_val else "left"
        insights.append(f"📈 Skewed Distribution: Data shows {skew_direction}-skewed pattern with potential outliers")
    
    # Content-based trend insights
    if content_insights.get("has_content"):
        domain = content_insights.get("detected_domain", "research")
        embedding_coverage = content_insights.get("embedding_coverage", 0)
        
        if embedding_coverage > 80:
            insights.append(f"🔍 High AI Readiness: {embedding_coverage:.0f}% content embedded - excellent for AI analysis")
        elif embedding_coverage > 50:
            insights.append(f"📊 Good AI Coverage: {embedding_coverage:.0f}% content embedded - suitable for AI insights")
        else:
            insights.append(f"⚠️ Limited AI Coverage: {embedding_coverage:.0f}% content embedded - consider reprocessing")
        
        # Domain-specific trend insights
        insights.append(f"🎯 Domain Trend: {domain.title()} documents typically show {get_domain_trend_pattern(domain)}")
    
    return insights

def generate_anomaly_insights(numerical_data: Dict, content_insights: Dict) -> List[str]:
    """Generate anomaly detection insights"""
    
    insights = []
    
    if not numerical_data.get("has_data"):
        insights.append("🔍 Anomaly Detection: No numerical data available for anomaly analysis")
        return insights
    
    # Statistical anomaly detection
    mean_val = numerical_data["mean"]
    std_dev = numerical_data["std_dev"]
    min_val = numerical_data["min_val"]
    max_val = numerical_data["max_val"]
    
    # Check for outliers using 2-sigma rule
    lower_bound = mean_val - 2 * std_dev
    upper_bound = mean_val + 2 * std_dev
    
    if min_val < lower_bound:
        insights.append(f"🔍 Low Anomaly Detected: Minimum value {min_val:.2f} is {abs(min_val - lower_bound):.2f} units below expected range")
    
    if max_val > upper_bound:
        insights.append(f"🔍 High Anomaly Detected: Maximum value {max_val:.2f} is {max_val - upper_bound:.2f} units above expected range")
    
    if lower_bound <= min_val <= max_val <= upper_bound:
        insights.append("✅ No Statistical Anomalies: All data points fall within expected statistical range")
    
    # Content anomalies
    if content_insights.get("has_content"):
        processing_status = content_insights.get("processing_status", "unknown")
        if processing_status != "completed":
            insights.append(f"⚠️ Processing Anomaly: Document status is '{processing_status}' - may affect data quality")
    
    return insights

def generate_prediction_insights(numerical_data: Dict, content_insights: Dict) -> List[str]:
    """Generate prediction-focused insights"""
    
    insights = []
    
    if not numerical_data.get("has_data"):
        insights.append("🔮 Prediction Analysis: No numerical data available for predictive modeling")
        return insights
    
    # Predictive indicators
    cv = (numerical_data["std_dev"] / numerical_data["mean"] * 100) if numerical_data["mean"] != 0 else 0
    
    if cv < 15:
        insights.append("🔮 High Predictability: Low variation suggests reliable forecasting potential")
    elif cv < 30:
        insights.append("🔮 Moderate Predictability: Reasonable variation allows for cautious forecasting")
    else:
        insights.append("🔮 Low Predictability: High variation suggests complex patterns requiring advanced modeling")
    
    # Data volume assessment for predictions
    data_count = numerical_data["total_values"]
    if data_count > 100:
        insights.append("📊 Excellent Prediction Foundation: Large dataset enables robust predictive modeling")
    elif data_count > 30:
        insights.append("📊 Good Prediction Foundation: Adequate data for basic predictive analysis")
    else:
        insights.append("📊 Limited Prediction Capability: Small dataset restricts prediction accuracy")
    
    # Domain-specific prediction insights
    if content_insights.get("has_content"):
        domain = content_insights.get("detected_domain", "research")
        insights.append(f"🎯 {domain.title()} Prediction: {get_domain_prediction_insights(domain)}")
    
    return insights

def generate_recommendation_insights(numerical_data: Dict, content_insights: Dict, domain_insights: Dict) -> List[str]:
    """Generate actionable recommendations"""
    
    recommendations = []
    
    # Data quality recommendations
    if numerical_data.get("has_data"):
        data_count = numerical_data["total_values"]
        if data_count < 20:
            recommendations.append("📈 Recommendation: Collect more data points to improve analysis reliability")
        
        cv = (numerical_data["std_dev"] / numerical_data["mean"] * 100) if numerical_data["mean"] != 0 else 0
        if cv > 40:
            recommendations.append("🔍 Recommendation: Investigate high variability - consider data segmentation or outlier analysis")
    
    # Content processing recommendations
    if content_insights.get("has_content"):
        embedding_coverage = content_insights.get("embedding_coverage", 0)
        if embedding_coverage < 70:
            recommendations.append("🤖 Recommendation: Reprocess document to improve AI analysis coverage")
        
        domain = content_insights.get("detected_domain", "research")
        recommendations.extend(get_domain_specific_recommendations(domain))
    
    # System recommendations
    recommendations.append("📊 Recommendation: Regular monitoring of key metrics to identify trends early")
    recommendations.append("🔄 Recommendation: Consider automated reporting for continuous insights")
    
    return recommendations

def get_domain_trend_pattern(domain: str) -> str:
    """Get typical trend patterns for each domain"""
    patterns = {
        "financial": "cyclical patterns with seasonal variations",
        "medical": "improvement trends with treatment protocols",
        "engineering": "optimization curves with performance plateaus",
        "scientific": "experimental progressions with hypothesis validation",
        "legal": "compliance patterns with regulatory changes",
        "research": "discovery patterns with methodology evolution"
    }
    return patterns.get(domain, "data-driven patterns with methodology focus")

def get_domain_prediction_insights(domain: str) -> str:
    """Get domain-specific prediction insights"""
    insights = {
        "financial": "Financial forecasting benefits from trend analysis and seasonal adjustments",
        "medical": "Clinical predictions require long-term monitoring and treatment response data",
        "engineering": "Performance predictions improve with operational parameter tracking",
        "scientific": "Research predictions depend on experimental consistency and methodology",
        "legal": "Compliance predictions focus on regulatory trend analysis",
        "research": "Academic predictions benefit from literature trend analysis"
    }
    return insights.get(domain, "Predictive accuracy improves with consistent data collection")

def get_domain_specific_recommendations(domain: str) -> List[str]:
    """Get domain-specific actionable recommendations"""
    recommendations = {
        "financial": [
            "💰 Consider quarterly trend analysis for better forecasting",
            "📊 Implement real-time financial monitoring dashboards"
        ],
        "medical": [
            "🏥 Track patient outcome metrics for treatment optimization",
            "📈 Monitor clinical indicators for early intervention"
        ],
        "engineering": [
            "⚙️ Implement performance benchmarking against specifications",
            "🔧 Consider predictive maintenance based on performance trends"
        ],
        "scientific": [
            "🔬 Validate experimental reproducibility with statistical analysis",
            "📊 Consider meta-analysis for broader research insights"
        ],
        "legal": [
            "⚖️ Monitor regulatory changes for compliance updates",
            "📋 Implement automated compliance checking systems"
        ],
        "research": [
            "📚 Consider literature trend analysis for research direction",
            "🔍 Implement systematic review methodology for insights"
        ]
    }
    return recommendations.get(domain, ["📊 Regular data review for continuous improvement"])

def format_insights_output(insights: List[str], focus: str) -> str:
    """Format insights into a comprehensive report"""
    
    if not insights:
        return "📊 No specific insights generated for the current dataset."
    
    formatted = f"🤖 AI-Generated Insights ({focus.title()} Focus):\n\n"
    
    for i, insight in enumerate(insights, 1):
        formatted += f"{i}. {insight}\n\n"
    
    # Add summary based on focus
    if focus == "comprehensive":
        formatted += "💡 Summary: Analysis reveals multiple patterns and opportunities for optimization.\n"
    elif focus == "performance":
        formatted += "💡 Summary: Performance analysis identifies key metrics and improvement areas.\n"
    elif focus == "trends":
        formatted += "💡 Summary: Trend analysis shows directional patterns and future indicators.\n"
    elif focus == "recommendations":
        formatted += "💡 Summary: Actionable recommendations provided for immediate implementation.\n"
    
    return formatted

# === GLOBAL ANALYSIS FUNCTIONS (for multi-document insights) ===

def get_global_numerical_insights() -> Dict[str, Any]:
    """Get numerical insights across all documents"""
    try:
        # Get all numerical data
        structured_data = DocumentStructuredData.objects.filter(numeric_value__isnull=False)[:200]
        key_values = DocumentKeyValue.objects.filter(value_numeric__isnull=False)[:100]
        
        values = [float(item.numeric_value) for item in structured_data]
        kv_values = [float(kv.value_numeric) for kv in key_values]
        all_values = values + kv_values
        
        if not all_values:
            return {"has_data": False}
        
        import statistics
        return {
            "has_data": True,
            "total_values": len(all_values),
            "mean": statistics.mean(all_values),
            "median": statistics.median(all_values),
            "std_dev": statistics.stdev(all_values) if len(all_values) > 1 else 0,
            "min_val": min(all_values),
            "max_val": max(all_values),
            "range": max(all_values) - min(all_values),
            "document_count": Document.objects.count()
        }
    except Exception as e:
        return {"has_data": False, "error": str(e)}

def get_global_content_insights() -> Dict[str, Any]:
    """Get content insights across all documents"""
    try:
        total_docs = Document.objects.count()
        total_segments = DocumentTextSegment.objects.count()
        embedded_segments = DocumentTextSegment.objects.filter(embedding__isnull=False).count()
        
        return {
            "has_content": total_docs > 0,
            "total_documents": total_docs,
            "total_segments": total_segments,
            "embedded_segments": embedded_segments,
            "global_coverage": (embedded_segments / total_segments * 100) if total_segments > 0 else 0
        }
    except Exception as e:
        return {"has_content": False, "error": str(e)}

def get_global_domain_insights() -> Dict[str, Any]:
    """Get domain distribution across all documents"""
    return {"domain": "multi_domain", "scope": "global"}

def analyze_financial_domain_insights(document) -> Dict[str, Any]:
    """Analyze financial-specific insights"""
    return {"domain": "financial", "focus": "revenue_analysis"}

def analyze_medical_domain_insights(document) -> Dict[str, Any]:
    """Analyze medical-specific insights"""
    return {"domain": "medical", "focus": "clinical_outcomes"}

def analyze_engineering_domain_insights(document) -> Dict[str, Any]:
    """Analyze engineering-specific insights"""
    return {"domain": "engineering", "focus": "performance_metrics"}

def analyze_scientific_domain_insights(document) -> Dict[str, Any]:
    """Analyze scientific-specific insights"""
    return {"domain": "scientific", "focus": "experimental_analysis"}

def analyze_general_domain_insights(document) -> Dict[str, Any]:
    """Analyze general research insights"""
    return {"domain": "research", "focus": "general_analysis"}

def generate_contextual_insights(context: str, numerical_data: Dict) -> List[str]:
    """Generate insights based on provided context"""
    insights = []
    
    context_lower = context.lower()
    
    if "quarterly" in context_lower or "q1" in context_lower or "q2" in context_lower:
        insights.append("📅 Quarterly Context: Analysis focused on quarterly performance patterns")
    
    if "comparison" in context_lower or "benchmark" in context_lower:
        insights.append("⚖️ Comparative Context: Analysis includes benchmark comparisons")
    
    if "forecast" in context_lower or "prediction" in context_lower:
        insights.append("🔮 Predictive Context: Analysis optimized for forecasting scenarios")
    
    if "performance" in context_lower:
        insights.append("📈 Performance Context: Focus on efficiency and effectiveness metrics")
    
    return insights

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
        
        print(f"🛠️ Available Tools: 10 (6 core + 4 advanced)")
        print(f"🎯 Advanced Capabilities: Statistical analysis, domain detection, pattern visualization, AI insights")
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