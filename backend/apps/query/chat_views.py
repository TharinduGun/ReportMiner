import time
import logging
from typing import List, Dict, Any
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from .rag_engine import get_rag_engine
from .mcp_server import (
    handle_extract_numerical_data,
    handle_calculate_metrics,
    handle_domain_analysis,
    handle_visualize_patterns,
    handle_generate_insights
)

from apps.ingestion.models import Document
from apps.ingestion.views import EnhancedFileUploadView 
logger = logging.getLogger(__name__)

class ChatQueryView(APIView):
    """
    Main chat endpoint that orchestrates RAG + MCP tools
    This is what the frontend will call for all chat interactions
    """
    
    def post(self, request):
        """
        Process a chat query with RAG retrieval and MCP tools
        
        Expected request body:
        {
            "question": "What financial data is in my documents?",
            "include_tools": true,
            "include_sources": true
        }
        """
        try:
            # Extract parameters from request
            question = request.data.get('question', '')
            include_tools = request.data.get('include_tools', True)
            include_sources = request.data.get('include_sources', True)
            
            if not question.strip():
                return Response({
                    'success': False,
                    'error': 'Question cannot be empty'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            start_time = time.time()
            
            # STEP 1: RAG Query for base answer
            rag_engine = get_rag_engine()
            rag_result = rag_engine.query(question, include_sources=include_sources)
            
            if not rag_result['success']:
                return Response({
                    'success': False,
                    'error': 'Failed to retrieve relevant documents'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # STEP 2: Analyze question to determine relevant tools
            tool_results = []
            if include_tools and rag_result['success']:
                relevant_tools = self.analyze_question_for_tools(question)
                
                # STEP 3: Execute relevant MCP tools
                for tool_name in relevant_tools:
                    tool_result = self.execute_mcp_tool_safely(tool_name, {
                        'question': question,
                        'context': rag_result['answer'],
                        'sources': rag_result.get('sources', [])
                    })
                    if tool_result:
                        tool_results.append(tool_result)
            
            # STEP 4: Prepare comprehensive response
            processing_time = time.time() - start_time
            
            return Response({
                'success': True,
                'answer': rag_result['answer'],
                'tool_results': tool_results,
                'sources': rag_result.get('sources', []) if include_sources else [],
                'metadata': {
                    'tools_used': len(tool_results),
                    'sources_found': len(rag_result.get('sources', [])),
                    'processing_time': round(processing_time, 2),
                    'response_type': 'enhanced' if tool_results else 'basic'
                }
            })
            
        except Exception as e:
            logger.error(f"Chat query failed: {str(e)}")
            return Response({
                'success': False,
                'error': f'Internal server error: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def analyze_question_for_tools(self, question: str) -> List[str]:
        """
        Determine which MCP tools are relevant for the question
        This is keyword-based analysis - you can make it smarter later
        """
        question_lower = question.lower()
        relevant_tools = []
        
        # Keywords that suggest specific tools
        if any(word in question_lower for word in ['data', 'numbers', 'values', 'extract', 'statistics']):
            relevant_tools.append('extract_numerical_data')
        
        if any(word in question_lower for word in ['calculate', 'metrics', 'average', 'trend', 'analysis']):
            relevant_tools.append('calculate_metrics')
        
        if any(word in question_lower for word in ['domain', 'type', 'classification', 'analyze', 'what kind']):
            relevant_tools.append('domain_analysis')
        
        if any(word in question_lower for word in ['visualize', 'chart', 'graph', 'pattern', 'show']):
            relevant_tools.append('visualize_patterns')
        
        if any(word in question_lower for word in ['insights', 'recommendations', 'summary', 'conclude']):
            relevant_tools.append('generate_insights')
        
        # Limit to 3 tools max for performance
        return relevant_tools[:3]
    
    def execute_mcp_tool_safely(self, tool_name: str, args: dict) -> dict:
        """
        Execute specific MCP tool with error handling
        Returns formatted result or None if failed
        """
        try:
            start_time = time.time()
            
            # Map tool names to handler functions
            tool_handlers = {
                'extract_numerical_data': handle_extract_numerical_data,
                'calculate_metrics': handle_calculate_metrics,
                'domain_analysis': handle_domain_analysis,
                'visualize_patterns': handle_visualize_patterns,
                'generate_insights': handle_generate_insights
            }
            
            if tool_name not in tool_handlers:
                logger.warning(f"Unknown tool requested: {tool_name}")
                return None
            
            # Execute the tool
            handler = tool_handlers[tool_name]
            result = handler(args)
            execution_time = time.time() - start_time
            
            return {
                'tool_name': tool_name,
                'result': result[0].text if result and len(result) > 0 else 'No result generated',
                'execution_time': round(execution_time, 2),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"MCP tool {tool_name} failed: {str(e)}")
            return {
                'tool_name': tool_name,
                'result': f'Tool execution failed: {str(e)}',
                'execution_time': 0,
                'success': False
            }


class ChatUploadView(APIView):
    """
    Handle file uploads with full processing pipeline including embeddings
    """
    
    def post(self, request):
        try:
            if 'file' not in request.FILES:
                return Response({
                    'success': False,
                    'error': 'No file provided'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            file = request.FILES['file']
            
            # Get or generate required fields
            filename = request.data.get('filename', file.name.split('.')[0])
            file_extension = file.name.split('.')[-1].lower()
            file_type = request.data.get('type', file_extension)
            
            # Create FileUpload record first
            from apps.ingestion.models import FileUpload
            from apps.ingestion.serializers import FileUploadSerializer
            
            upload_data = {
                'filename': filename,
                'file': file,
                'type': file_type
            }
            
            serializer = FileUploadSerializer(data=upload_data)
            if not serializer.is_valid():
                return Response({
                    'success': False,
                    'error': f'Validation failed: {serializer.errors}'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Save the file upload
            file_upload = serializer.save()
            
            # NOW USE THE ENHANCED PIPELINE FOR FULL PROCESSING
            from apps.ingestion.enhanced_upload_pipeline import EnhancedUploadPipeline
            
            pipeline = EnhancedUploadPipeline()
            result = pipeline.process_uploaded_file(file_upload)
            
            if result['success']:
                return Response({
                    'success': True,
                    'message': 'Document uploaded and processed successfully',
                    'document_id': result['document_id'],  # ✅ Real UUID from Document table
                    'filename': filename,
                    'file_type': file_type,
                    'processing_status': 'completed',
                    'upload_id': file_upload.id,
                    'text_preview': result.get('text_preview', ''),
                    'processing_results': result.get('processing_results', {}),
                    'embeddings_created': result['processing_results'].get('embeddings_created', 0)
                })
            else:
                return Response({
                    'success': False,
                    'error': f"Processing failed: {result.get('errors', ['Unknown error'])}",
                    'document_id': result.get('document_id'),
                    'upload_id': file_upload.id
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
        except Exception as e:
            logger.error(f"Chat upload failed: {str(e)}")
            return Response({
                'success': False,
                'error': f'Upload error: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ChatDocumentListView(APIView):
    """
    List documents for chat interface
    """
    
    def get(self, request):
        """
        Get list of user's documents
        """
        try:
            from apps.ingestion.models import Document
            
            documents = Document.objects.all().order_by('-uploaded_at')[:20]
            
            document_list = []
            for doc in documents:
                document_list.append({
                    'id': str(doc.id),
                    'filename': doc.filename,
                    'file_type': doc.file_type,
                    'uploaded_at': doc.uploaded_at.isoformat(),
                    'processing_status': doc.processing_status,
                    'text_segments_count': doc.text_segments.count() if hasattr(doc, 'text_segments') else 0
                })
            
            return Response({
                'success': True,
                'documents': document_list,
                'total_count': len(document_list)
            })
            
        except Exception as e:
            logger.error(f"Document list failed: {str(e)}")
            return Response({
                'success': False,
                'error': f'Failed to retrieve documents: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)