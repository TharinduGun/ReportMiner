"""
Test the final working MCP server functions
"""
import os
import sys
import asyncio

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'reportminer.settings')

import django
django.setup()

# Import the final MCP functions
from apps.query.mcp_server_final import (
    sync_test_connection, 
    sync_list_recent_documents, 
    sync_search_documents, 
    sync_query_natural_language,
    sync_get_document_summary
)

async def test_all_mcp_tools():
    """Test all MCP tools with async/sync handling"""
    print("🧪 Testing Final MCP Server Functions\n")
    
    loop = asyncio.get_running_loop()
    
    # Test 1: Connection test
    print("=== Test 1: System Connection ===")
    try:
        result = await loop.run_in_executor(None, sync_test_connection, {})
        print(f"✅ {result[0].text[:200]}...")
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
    
    # Test 2: List documents
    print("\n=== Test 2: List Recent Documents ===")
    try:
        result = await loop.run_in_executor(None, sync_list_recent_documents, {"limit": 3})
        print(f"✅ {result[0].text[:200]}...")
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
    
    # Test 3: Search documents
    print("\n=== Test 3: Search Documents ===")
    try:
        result = await loop.run_in_executor(None, sync_search_documents, {"query": "report", "limit": 2})
        print(f"✅ {result[0].text[:200]}...")
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
    
    # Test 4: Natural language query
    print("\n=== Test 4: Natural Language Query ===")
    try:
        result = await loop.run_in_executor(None, sync_query_natural_language, {
            "question": "What types of documents do I have?", 
            "include_sources": True
        })
        print(f"✅ {result[0].text[:200]}...")
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
    
    print("\n🎉 All MCP tool tests completed!")
    print("💡 If all tests passed, your MCP tools are working correctly!")

if __name__ == "__main__":
    asyncio.run(test_all_mcp_tools())