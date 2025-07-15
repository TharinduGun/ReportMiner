"""
Test MCP functions independently to verify async/sync fix
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

# Import the MCP functions
from apps.query.mcp_server import handle_search_documents, handle_list_recent_documents, handle_query_natural_language

async def test_mcp_functions():
    """Test MCP functions with async/sync handling"""
    print("🧪 Testing MCP Functions with Async/Sync Fix\n")
    
    loop = asyncio.get_running_loop()
    
    # Test 1: List documents
    print("=== Test 1: List Recent Documents ===")
    try:
        result = await loop.run_in_executor(None, handle_list_recent_documents, {"limit": 5})
        print(f"✅ SUCCESS: {result[0].text[:100]}...")
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
    
    print("\n=== Test 2: Search Documents ===")
    try:
        result = await loop.run_in_executor(None, handle_search_documents, {"query": "financial", "limit": 3})
        print(f"✅ SUCCESS: {result[0].text[:100]}...")
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
    
    print("\n=== Test 3: Natural Language Query ===")
    try:
        result = await loop.run_in_executor(None, handle_query_natural_language, {"question": "What documents do I have?", "include_sources": True})
        print(f"✅ SUCCESS: {result[0].text[:100]}...")
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
    
    print("\n🎉 All MCP functions tested!")

if __name__ == "__main__":
    asyncio.run(test_mcp_functions())