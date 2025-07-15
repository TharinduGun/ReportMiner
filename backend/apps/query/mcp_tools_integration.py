"""
MCP Tools Integration for Django Views
Since the MCP server has library compatibility issues, 
this integrates the working MCP functions directly into Django
"""
import asyncio
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from django.http import JsonResponse
from django.views import View
import json

# Import our working MCP functions
from .mcp_server_final import (
    sync_test_connection,
    sync_list_recent_documents, 
    sync_search_documents,
    sync_query_natural_language,
    sync_get_document_summary
)

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=4)

class MCPToolsView(View):
    """
    Django view that provides MCP-like tools functionality
    Integrates the working MCP functions into your Django API
    """
    
    async def post(self, request):
        """Handle MCP tool requests"""
        try:
            data = json.loads(request.body)
            tool_name = data.get('tool')
            arguments = data.get('arguments', {})
            
            if not tool_name:
                return JsonResponse({
                    'success': False,
                    'error': 'Tool name is required'
                }, status=400)
            
            # Execute the tool in thread pool
            loop = asyncio.get_running_loop()
            result = await self._execute_tool(loop, tool_name, arguments)
            
            return JsonResponse({
                'success': True,
                'tool': tool_name,
                'result': result
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    
    async def _execute_tool(self, loop, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute MCP tool in thread pool"""
        
        if tool_name == "test_connection":
            result = await loop.run_in_executor(executor, sync_test_connection, arguments)
        elif tool_name == "list_recent_documents":
            result = await loop.run_in_executor(executor, sync_list_recent_documents, arguments)
        elif tool_name == "search_documents":
            result = await loop.run_in_executor(executor, sync_search_documents, arguments)
        elif tool_name == "query_natural_language":
            result = await loop.run_in_executor(executor, sync_query_natural_language, arguments)
        elif tool_name == "get_document_summary":
            result = await loop.run_in_executor(executor, sync_get_document_summary, arguments)
        else:
            return f"❌ Unknown tool: {tool_name}"
        
        # Extract text from TextContent objects
        if result and hasattr(result[0], 'text'):
            return result[0].text
        else:
            return str(result)

# Sync functions for direct use in regular Django views
def execute_mcp_tool_sync(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Synchronous wrapper for MCP tools
    Use this in regular Django views
    """
    try:
        if tool_name == "test_connection":
            result = sync_test_connection(arguments)
        elif tool_name == "list_recent_documents":
            result = sync_list_recent_documents(arguments)
        elif tool_name == "search_documents":
            result = sync_search_documents(arguments)
        elif tool_name == "query_natural_language":
            result = sync_query_natural_language(arguments)
        elif tool_name == "get_document_summary":
            result = sync_get_document_summary(arguments)
        else:
            return f"❌ Unknown tool: {tool_name}"
        
        # Extract text from TextContent objects
        if result and hasattr(result[0], 'text'):
            return result[0].text
        else:
            return str(result)
            
    except Exception as e:
        return f"❌ Tool execution failed: {str(e)}"

# Enhanced chat view with MCP tools integration
def enhanced_chat_query_with_mcp_tools(question: str, use_tools: bool = True) -> Dict[str, Any]:
    """
    Enhanced chat query that can optionally use MCP tools
    Use this to integrate MCP functionality into your chat system
    """
    
    # Determine if we should use specific tools based on the question
    tools_to_use = []
    
    if use_tools:
        question_lower = question.lower()
        
        # Smart tool selection based on question content
        if any(word in question_lower for word in ['list', 'show me', 'what documents', 'recent']):
            tools_to_use.append('list_recent_documents')
        
        if any(word in question_lower for word in ['search', 'find', 'look for']):
            tools_to_use.append('search_documents')
        
        if any(word in question_lower for word in ['summary', 'details', 'information about']):
            tools_to_use.append('get_document_summary')
        
        # Always try natural language query for comprehensive answers
        tools_to_use.append('query_natural_language')
    
    results = []
    
    # Execute selected tools
    for tool in tools_to_use:
        try:
            if tool == 'list_recent_documents':
                result = execute_mcp_tool_sync(tool, {"limit": 5})
                results.append({"tool": tool, "result": result})
                
            elif tool == 'search_documents':
                # Extract search terms from question
                search_terms = question_lower.replace('search', '').replace('find', '').strip()
                if search_terms:
                    result = execute_mcp_tool_sync(tool, {"query": search_terms, "limit": 3})
                    results.append({"tool": tool, "result": result})
                    
            elif tool == 'query_natural_language':
                result = execute_mcp_tool_sync(tool, {"question": question, "include_sources": True})
                results.append({"tool": tool, "result": result})
                
        except Exception as e:
            results.append({"tool": tool, "error": str(e)})
    
    # Combine results into a comprehensive response
    if results:
        # Primary answer from natural language query
        main_answer = ""
        additional_info = []
        
        for result in results:
            if result["tool"] == "query_natural_language" and "result" in result:
                main_answer = result["result"]
            elif "result" in result:
                additional_info.append(f"**{result['tool']}**: {result['result'][:200]}...")
        
        response = main_answer
        if additional_info:
            response += "\n\n**Additional Information:**\n" + "\n".join(additional_info)
            
        return {
            "success": True,
            "answer": response,
            "tools_used": [r["tool"] for r in results if "result" in r],
            "tool_results": results
        }
    else:
        # Fallback to basic natural language query
        try:
            result = execute_mcp_tool_sync("query_natural_language", {"question": question, "include_sources": True})
            return {
                "success": True,
                "answer": result,
                "tools_used": ["query_natural_language"],
                "tool_results": [{"tool": "query_natural_language", "result": result}]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "answer": "I encountered an error processing your question."
            }

# Available tools info
AVAILABLE_MCP_TOOLS = {
    "test_connection": {
        "description": "Test database and system connectivity",
        "parameters": {}
    },
    "list_recent_documents": {
        "description": "List recently uploaded documents",
        "parameters": {"limit": "number of documents (default: 10)"}
    },
    "search_documents": {
        "description": "Search documents using vector similarity",
        "parameters": {"query": "search query", "limit": "max results (default: 5)"}
    },
    "query_natural_language": {
        "description": "Ask natural language questions about documents",
        "parameters": {"question": "your question", "include_sources": "include source docs (default: true)"}
    },
    "get_document_summary": {
        "description": "Get detailed information about a specific document",
        "parameters": {"document_id": "UUID of the document"}
    }
}