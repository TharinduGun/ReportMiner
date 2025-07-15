"""
Simple MCP Server for ReportMiner - ASYNC/SYNC FIXED
Minimal implementation that actually works
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
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# MCP imports (after Django setup)
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server

# Django imports (after setup)
from django.db import connection, connections
from apps.ingestion.models import Document, DocumentTextSegment

# Create MCP server instance
server = Server("reportminer-simple")

# Thread pool for database operations
executor = ThreadPoolExecutor(max_workers=2)

@server.list_tools()
async def list_tools() -> List[Tool]:
    """Define core MCP tools"""
    return [
        Tool(
            name="list_documents",
            description="List all documents in the system",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max documents", "default": 10}
                }
            }
        ),
        
        Tool(
            name="search_simple",
            description="Simple document search",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Execute tools with proper async/sync handling"""
    try:
        print(f"🔧 Tool called: {name}")
        
        # Run database operations in thread pool
        loop = asyncio.get_running_loop()
        
        if name == "list_documents":
            return await loop.run_in_executor(executor, sync_list_documents, arguments)
        elif name == "search_simple":
            return await loop.run_in_executor(executor, sync_search_simple, arguments)
        else:
            return [TextContent(type="text", text=f"❌ Unknown tool: {name}")]
            
    except Exception as e:
        error_msg = f"❌ Error in {name}: {str(e)}"
        print(error_msg)
        return [TextContent(type="text", text=error_msg)]

def sync_list_documents(args: Dict[str, Any]) -> List[TextContent]:
    """List documents - pure sync function"""
    try:
        # Close any existing connections
        connections.close_all()
        
        limit = args.get("limit", 10)
        
        # Use raw SQL to avoid Django ORM async issues
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT filename, file_type, processing_status, uploaded_at 
                FROM documents 
                ORDER BY uploaded_at DESC 
                LIMIT %s
            """, [limit])
            
            rows = cursor.fetchall()
        
        if not rows:
            return [TextContent(type="text", text="📁 No documents found")]
        
        results = [f"📚 Found {len(rows)} documents:\n"]
        
        for i, (filename, file_type, status, uploaded_at) in enumerate(rows, 1):
            status_emoji = "✅" if status == "completed" else "🔄" if status == "processing" else "⏳"
            results.append(f"{i}. {status_emoji} {filename} ({file_type}) - {status}")
        
        return [TextContent(type="text", text="\n".join(results))]
        
    except Exception as e:
        return [TextContent(type="text", text=f"❌ List error: {str(e)}")]

def sync_search_simple(args: Dict[str, Any]) -> List[TextContent]:
    """Simple search - pure sync function"""
    try:
        # Close any existing connections
        connections.close_all()
        
        query = args.get("query", "")
        
        if not query:
            return [TextContent(type="text", text="❌ Query cannot be empty")]
        
        # Use raw SQL for simple text search
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT d.filename, ts.content, ts.sequence_number
                FROM documents d
                JOIN document_text_segments ts ON d.id = ts.document_id
                WHERE ts.content ILIKE %s
                LIMIT 5
            """, [f"%{query}%"])
            
            rows = cursor.fetchall()
        
        if not rows:
            return [TextContent(type="text", text=f"🔍 No results found for: {query}")]
        
        results = [f"🔍 Found {len(rows)} results for '{query}':\n"]
        
        for i, (filename, content, sequence) in enumerate(rows, 1):
            preview = content[:100] + "..." if len(content) > 100 else content
            results.append(f"{i}. **{filename}** (Segment {sequence})")
            results.append(f"   📄 \"{preview}\"")
            results.append("")
        
        return [TextContent(type="text", text="\n".join(results))]
        
    except Exception as e:
        return [TextContent(type="text", text=f"❌ Search error: {str(e)}")]

def check_system():
    """Check system status"""
    try:
        print("🚀 Starting Simple MCP Server...")
        
        # Simple database check
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM document_text_segments")
            segment_count = cursor.fetchone()[0]
        
        print(f"✅ Database: {doc_count} documents, {segment_count} segments")
        print(f"🛠️ Tools: list_documents, search_simple")
        print(f"🎯 Ready for connections...")
        
        return True
        
    except Exception as e:
        print(f"❌ System check failed: {e}")
        return False

async def main():
    """Main async function"""
    try:
        # Check system in main thread
        if not check_system():
            return
        
        # Start MCP server
        async with stdio_server(server) as streams:
            await server.run(*streams)
            
    except Exception as e:
        print(f"❌ Server failed: {e}")
        raise
    finally:
        executor.shutdown(wait=True)

if __name__ == "__main__":
    asyncio.run(main())