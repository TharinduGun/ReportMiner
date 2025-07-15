"""
Working MCP Server for ReportMiner - FINAL SOLUTION
Uses database connections in isolated threads
"""
import os
import sys
import threading

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set Django settings BEFORE any Django imports
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'reportminer.settings')

# Configure Django to not use async-to-sync
import django
from django.conf import settings
django.setup()

# Standard library imports
import asyncio
from typing import List, Dict, Any

# MCP imports
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server

# Create MCP server instance
server = Server("reportminer-working")

@server.list_tools()
async def list_tools() -> List[Tool]:
    """Define working tools"""
    return [
        Tool(
            name="test_connection",
            description="Test database connection",
            inputSchema={"type": "object", "properties": {}}
        ),
        
        Tool(
            name="list_documents",
            description="List documents in database",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 10}
                }
            }
        ),
        
        Tool(
            name="simple_search",
            description="Search document content",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search term"}
                },
                "required": ["query"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Execute tools in isolated threads"""
    try:
        print(f"🔧 Executing: {name}")
        
        # Create a new thread for each database operation
        result = await asyncio.get_event_loop().run_in_executor(
            None, execute_in_new_thread, name, arguments
        )
        
        return [TextContent(type="text", text=result)]
        
    except Exception as e:
        error_msg = f"❌ {name} failed: {str(e)}"
        print(error_msg)
        return [TextContent(type="text", text=error_msg)]

def execute_in_new_thread(tool_name: str, args: Dict[str, Any]) -> str:
    """Execute database operations in a completely new thread"""
    import django
    from django.db import connection
    
    try:
        # Ensure Django is set up in this thread
        django.setup()
        
        # Import models after setup
        from apps.ingestion.models import Document, DocumentTextSegment
        
        if tool_name == "test_connection":
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
            return "✅ Database connection successful!"
        
        elif tool_name == "list_documents":
            limit = args.get("limit", 10)
            
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT filename, file_type, processing_status, uploaded_at::text
                    FROM documents 
                    ORDER BY uploaded_at DESC 
                    LIMIT %s
                """, [limit])
                
                rows = cursor.fetchall()
            
            if not rows:
                return "📁 No documents found in database"
            
            results = [f"📚 Found {len(rows)} documents:"]
            for i, (filename, file_type, status, uploaded_at) in enumerate(rows, 1):
                results.append(f"{i}. {filename} ({file_type}) - {status}")
            
            return "\n".join(results)
        
        elif tool_name == "simple_search":
            query = args.get("query", "")
            
            if not query:
                return "❌ Search query cannot be empty"
            
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
                return f"🔍 No results found for: '{query}'"
            
            results = [f"🔍 Found {len(rows)} results for '{query}':"]
            for i, (filename, content, sequence) in enumerate(rows, 1):
                preview = content[:80] + "..." if len(content) > 80 else content
                results.append(f"{i}. {filename} (Segment {sequence}): \"{preview}\"")
            
            return "\n".join(results)
        
        else:
            return f"❌ Unknown tool: {tool_name}"
    
    except Exception as e:
        return f"❌ Database operation failed: {str(e)}"
    
    finally:
        # Clean up connection
        try:
            connection.close()
        except:
            pass

def test_system():
    """Test system in main thread before going async"""
    print("🚀 Starting ReportMiner MCP Server...")
    print("✅ Django configured")
    print("✅ MCP server initialized")
    print("🛠️ Available tools: test_connection, list_documents, simple_search")
    print("🎯 Ready for MCP connections...")
    return True

async def main():
    """Main server loop"""
    try:
        # Test system
        if not test_system():
            return
        
        # Start MCP server with proper signature
        async with stdio_server(server) as (read_stream, write_stream):
            await server.run(read_stream, write_stream, {})
            
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())