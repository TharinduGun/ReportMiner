#!/usr/bin/env python3
"""
Test script to reproduce the MCP integration error
"""

import os
import sys
import django
from pathlib import Path

# Add the Django project to the path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'reportminer.settings')
django.setup()

from apps.query.services import run_query

def test_mcp_tools():
    """Test the MCP integration with problematic queries"""
    
    print("=== Testing MCP Tool Integration ===")
    
    # Test 1: list available documents (reported to fail)
    print("\n1. Testing 'list_available_documents':")
    try:
        result = run_query("list_available_documents", use_mcp=True)
        print(f"✓ Success: {result['answer'][:100]}...")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 2: list available tools (reported to show error in logs but work)
    print("\n2. Testing 'list available tools':")
    try:
        result = run_query("list available tools", use_mcp=True)
        print(f"✓ Success: {result['answer'][:100]}...")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 3: get collection stats (should work)
    print("\n3. Testing 'get_collection_stats':")
    try:
        result = run_query("get collection stats", use_mcp=True)
        print(f"✓ Success: {result['answer'][:100]}...")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 4: Direct MCP tools loading
    print("\n4. Testing direct MCP tools loading:")
    try:
        from apps.query.services import get_mcp_tools
        tools = get_mcp_tools()
        print(f"✓ Loaded {len(tools)} MCP tools:")
        for tool in tools:
            print(f"  - {tool.name}: {getattr(tool, 'description', 'No description')}")
            print(f"    Args schema: {getattr(tool, 'args_schema', 'None')}")
    except Exception as e:
        print(f"✗ Error loading MCP tools: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mcp_tools()