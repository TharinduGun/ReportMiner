#!/usr/bin/env python3
"""
Script to inspect the actual MCP tools and their schemas
"""

import os
import sys
import django
import asyncio
import json
from pathlib import Path

# Add the Django project to the path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'reportminer.settings')
django.setup()

async def inspect_mcp_tools():
    """Inspect MCP tools directly from the server"""
    print("=== Direct MCP Server Tool Inspection ===")
    
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.sessions import create_session
        
        # Create connection config
        mcp_server_path = backend_dir / "mcp_server.py"
        connection = {
            "command": "python",
            "args": [str(mcp_server_path)],
            "transport": "stdio"
        }
        
        print(f"Connecting to MCP server at: {mcp_server_path}")
        
        # Create a session directly
        async with create_session(connection) as session:
            await session.initialize()
            
            # List tools
            print("\n=== Listing Tools ===")
            list_tools_result = await session.list_tools()
            print(f"Found {len(list_tools_result.tools)} tools")
            
            for i, tool in enumerate(list_tools_result.tools):
                print(f"\n--- Tool {i+1}: {tool.name} ---")
                print(f"Description: {tool.description}")
                print(f"Input Schema: {tool.inputSchema}")
                print(f"Input Schema type: {type(tool.inputSchema)}")
                
                # Pretty print the schema if it's a dict
                if isinstance(tool.inputSchema, dict):
                    print(f"Schema JSON:\n{json.dumps(tool.inputSchema, indent=2)}")
                elif tool.inputSchema is None:
                    print("Schema: None (no parameters)")
                else:
                    print(f"Schema: {tool.inputSchema}")
                
                # Check if schema has properties
                if isinstance(tool.inputSchema, dict):
                    properties = tool.inputSchema.get('properties', {})
                    required = tool.inputSchema.get('required', [])
                    print(f"Properties: {list(properties.keys())}")
                    print(f"Required: {required}")
                    print(f"Has parameters: {len(properties) > 0}")
    
    except Exception as e:
        print(f"Error inspecting MCP tools: {e}")
        import traceback
        traceback.print_exc()

async def test_langchain_conversion():
    """Test how the tools are converted to LangChain tools"""
    print("\n\n=== LangChain Tool Conversion Test ===")
    
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        
        # Create client
        mcp_server_path = backend_dir / "mcp_server.py"
        client = MultiServerMCPClient({
            "reportminer": {
                "command": "python",
                "args": [str(mcp_server_path)],
                "transport": "stdio"
            }
        })
        
        print("Getting tools via MultiServerMCPClient...")
        tools = await client.get_tools()
        print(f"Successfully loaded {len(tools)} LangChain tools")
        
        for i, tool in enumerate(tools):
            print(f"\n--- LangChain Tool {i+1}: {tool.name} ---")
            print(f"Description: {tool.description}")
            print(f"Args schema: {tool.args_schema}")
            print(f"Args schema type: {type(tool.args_schema)}")
            
            # Check if it's a pydantic model
            try:
                from pydantic import BaseModel
                if tool.args_schema and issubclass(tool.args_schema, BaseModel):
                    print("✓ Valid Pydantic BaseModel")
                    # Get model fields
                    if hasattr(tool.args_schema, 'model_fields'):
                        fields = tool.args_schema.model_fields
                        print(f"Model fields: {list(fields.keys())}")
                    elif hasattr(tool.args_schema, '__fields__'):
                        fields = tool.args_schema.__fields__
                        print(f"Fields: {list(fields.keys())}")
                elif tool.args_schema is None:
                    print("⚠ No args schema (None)")
                else:
                    print(f"✗ Not a Pydantic BaseModel: {type(tool.args_schema)}")
                    print(f"Args schema value: {tool.args_schema}")
            except Exception as e:
                print(f"Error checking args_schema: {e}")
    
    except Exception as e:
        print(f"Error testing LangChain conversion: {e}")
        import traceback
        traceback.print_exc()

async def main():
    await inspect_mcp_tools()
    await test_langchain_conversion()

if __name__ == "__main__":
    asyncio.run(main())