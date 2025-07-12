# File: backend/test_mcp_manual.py
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'reportminer.settings')
django.setup()

# Test the tool functions directly
import sys
sys.path.append('apps/query')

try:
    from apps.query.mcp_server import handle_list_recent_documents, handle_extract_numerical_data
    
    print("🧪 Testing MCP Tools Directly...")
    
    # Test 1: List documents
    result1 = handle_list_recent_documents({"limit": 3})
    print(f"✅ List documents: {len(result1)} results")
    print(f"📋 Preview: {result1[0].text[:100]}...")
    
    print("\n" + "="*50)
    
    # Test 2: Extract numerical data
    result2 = handle_extract_numerical_data({})
    print(f"✅ Extract data: {len(result2)} results")
    print(f"📊 Preview: {result2[0].text[:100]}...")
    
    print("\n🎉 All MCP tools are working!")
    
except Exception as e:
    print(f"❌ Tool test failed: {e}")