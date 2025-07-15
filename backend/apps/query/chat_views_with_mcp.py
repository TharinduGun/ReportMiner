"""
Enhanced Chat Views with Working MCP Tools Integration
This provides a working alternative to the MCP server
"""
import time
import logging
import json
from typing import Dict, Any
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

# Import our working MCP tools integration
from .mcp_tools_integration import enhanced_chat_query_with_mcp_tools, execute_mcp_tool_sync, AVAILABLE_MCP_TOOLS

logger = logging.getLogger(__name__)

class EnhancedChatQueryView(APIView):
    """
    Enhanced chat endpoint with working MCP tools integration
    This replaces the problematic MCP server with direct function calls
    """
    
    def post(self, request):
        """Enhanced chat with working MCP tools"""
        try:
            # Extract parameters
            question = request.data.get('question', '')
            use_mcp_tools = request.data.get('use_mcp_tools', True)
            session_id = request.data.get('session_id', f"session_{int(time.time())}")
            
            if not question.strip():
                return Response({
                    'success': False,
                    'message': 'Question cannot be empty',
                    'error_code': 'EMPTY_QUESTION',
                    'session_id': session_id,
                    'timestamp': int(time.time())
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Use enhanced chat with MCP tools
            start_time = time.time()
            result = enhanced_chat_query_with_mcp_tools(question, use_tools=use_mcp_tools)
            processing_time = time.time() - start_time
            
            if result.get('success', False):
                return Response({
                    'success': True,
                    'message': result.get('answer', ''),
                    'tools_used': result.get('tools_used', []),
                    'session_id': session_id,
                    'processing_time': round(processing_time, 2),
                    'timestamp': int(time.time()),
                    'metadata': {
                        'mcp_tools_enabled': use_mcp_tools,
                        'tools_executed': len(result.get('tools_used', [])),
                        'response_type': 'mcp_enhanced',
                        'model_used': 'gpt-4o'
                    }
                }, status=status.HTTP_200_OK)
            else:
                return Response({
                    'success': False,
                    'message': result.get('error', 'Unknown error occurred'),
                    'error_code': 'MCP_PROCESSING_ERROR',
                    'session_id': session_id,
                    'timestamp': int(time.time())
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
        except Exception as e:
            logger.error(f"Enhanced chat query failed: {str(e)}")
            return Response({
                'success': False,
                'message': 'I encountered an error processing your request.',
                'error_code': 'GENERAL_ERROR',
                'session_id': request.data.get('session_id', ''),
                'timestamp': int(time.time()),
                'error_details': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class MCPToolDirectView(APIView):
    """
    Direct MCP tool execution endpoint
    Allows calling specific MCP tools directly
    """
    
    def post(self, request):
        """Execute a specific MCP tool"""
        try:
            tool_name = request.data.get('tool')
            arguments = request.data.get('arguments', {})
            
            if not tool_name:
                return Response({
                    'success': False,
                    'error': 'Tool name is required'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            if tool_name not in AVAILABLE_MCP_TOOLS:
                return Response({
                    'success': False,
                    'error': f'Unknown tool: {tool_name}',
                    'available_tools': list(AVAILABLE_MCP_TOOLS.keys())
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Execute the tool
            start_time = time.time()
            result = execute_mcp_tool_sync(tool_name, arguments)
            processing_time = time.time() - start_time
            
            return Response({
                'success': True,
                'tool': tool_name,
                'result': result,
                'arguments': arguments,
                'processing_time': round(processing_time, 2),
                'timestamp': int(time.time())
            })
            
        except Exception as e:
            logger.error(f"MCP tool execution failed: {str(e)}")
            return Response({
                'success': False,
                'error': str(e),
                'timestamp': int(time.time())
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def get(self, request):
        """Get available MCP tools"""
        return Response({
            'success': True,
            'available_tools': AVAILABLE_MCP_TOOLS,
            'total_tools': len(AVAILABLE_MCP_TOOLS)
        })

class MCPSystemStatusView(APIView):
    """
    Check MCP system status
    """
    
    def get(self, request):
        """Get system status using MCP tools"""
        try:
            # Use the test_connection tool to check system status
            status_result = execute_mcp_tool_sync('test_connection', {})
            
            return Response({
                'success': True,
                'system_status': status_result,
                'tools_available': list(AVAILABLE_MCP_TOOLS.keys()),
                'mcp_integration': 'direct_function_calls',
                'timestamp': int(time.time())
            })
            
        except Exception as e:
            return Response({
                'success': False,
                'error': str(e),
                'timestamp': int(time.time())
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Function-based view for simple testing
@csrf_exempt
def test_mcp_tools(request):
    """Simple test endpoint for MCP tools"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            tool_name = data.get('tool', 'test_connection')
            arguments = data.get('arguments', {})
            
            result = execute_mcp_tool_sync(tool_name, arguments)
            
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
    
    # GET request - show available tools
    return JsonResponse({
        'available_tools': AVAILABLE_MCP_TOOLS,
        'example_usage': {
            'method': 'POST',
            'body': {
                'tool': 'test_connection',
                'arguments': {}
            }
        }
    })

# Integration with existing orchestrator
def get_mcp_enhanced_orchestrator():
    """
    Get an orchestrator that uses our working MCP tools
    This can replace your existing orchestrator
    """
    class MCPEnhancedOrchestrator:
        def process_query(self, question: str, document_ids=None):
            """Process query using working MCP tools"""
            try:
                result = enhanced_chat_query_with_mcp_tools(question, use_tools=True)
                
                # Format to match expected orchestrator output
                return {
                    'success': result.get('success', False),
                    'answer': result.get('answer', ''),
                    'sources': [],  # Can be enhanced to extract sources from tool results
                    'tool_results': result.get('tool_results', []),
                    'confidence': 0.85  # Can be calculated based on tool success
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'answer': 'I encountered an error processing your request.'
                }
    
    return MCPEnhancedOrchestrator()