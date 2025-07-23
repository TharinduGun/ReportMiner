#!/usr/bin/env python3
"""
Debug script to understand FastMCP tool definitions
"""

import inspect
from mcp.server.fastmcp import FastMCP

# Create a minimal MCP server to understand tool structure
mcp = FastMCP("test")

@mcp.tool()
def test_tool_with_params(param1: str, param2: int = 5) -> str:
    """A test tool with parameters"""
    return f"Got {param1} and {param2}"

@mcp.tool()
def test_tool_no_params() -> str:
    """A test tool with no parameters"""
    return "No params needed"

# Inspect the tools
print("=== FastMCP Tool Analysis ===")

# Get tools from the FastMCP instance
tools = mcp._tools if hasattr(mcp, '_tools') else {}

print(f"Found {len(tools)} tools in FastMCP instance")

for name, tool_func in tools.items():
    print(f"\nTool: {name}")
    print(f"  Function: {tool_func}")
    print(f"  Signature: {inspect.signature(tool_func)}")
    print(f"  Annotations: {tool_func.__annotations__}")
    print(f"  Doc: {tool_func.__doc__}")
    
    # Check if it has args_schema attribute
    print(f"  Has args_schema: {hasattr(tool_func, 'args_schema')}")
    if hasattr(tool_func, 'args_schema'):
        print(f"  args_schema: {tool_func.args_schema}")

# Also inspect the decorator itself
print(f"\n=== MCP Tool Decorator Analysis ===")
print(f"FastMCP class: {FastMCP}")
print(f"tool method: {FastMCP.tool}")

# Let's also test pydantic schema generation
try:
    from pydantic import BaseModel, Field
    from typing import get_type_hints
    
    print(f"\n=== Pydantic Schema Analysis ===")
    
    def analyze_function_schema(func):
        """Analyze how a function would be converted to Pydantic schema"""
        print(f"\nAnalyzing function: {func.__name__}")
        
        # Get type hints
        hints = get_type_hints(func)
        print(f"  Type hints: {hints}")
        
        # Get signature
        sig = inspect.signature(func)
        print(f"  Parameters: {list(sig.parameters.keys())}")
        
        # Check if function has no parameters
        has_params = len([p for p in sig.parameters.values() if p.name != 'return']) > 0
        print(f"  Has parameters: {has_params}")
        
        return has_params
    
    # Analyze our test tools
    analyze_function_schema(test_tool_with_params)
    analyze_function_schema(test_tool_no_params)
    
except Exception as e:
    print(f"Pydantic analysis error: {e}")