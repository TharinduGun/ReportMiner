import time
import logging
import re
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
from .llm_orchestrator import get_orchestrator
from apps.ingestion.models import Document
from apps.ingestion.views import EnhancedFileUploadView 

logger = logging.getLogger(__name__)

class ChatQueryView(APIView):
    """
    Main chat endpoint that orchestrates RAG + MCP tools
    This is what the frontend will call for all chat interactions
    
    MCP Tools are used through the LLM Orchestrator, not directly here.
    """
    
    def post(self, request):
        """Enhanced chat with LLM orchestration"""
        try:
            # Extract parameters
            question = request.data.get('question', '')
            include_tools = request.data.get('include_tools', True)
            include_sources = request.data.get('include_sources', True)
            session_id = request.data.get('session_id', f"session_{int(time.time())}")
                                          
            if not question.strip():
                return Response({
                    'success': False,
                    'message': 'Question cannot be empty',
                    'error_code': 'EMPTY_QUESTION',
                    'session_id': session_id,
                    'timestamp': int(time.time())
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Use intelligent orchestrator (this calls MCP tools intelligently)
            orchestrator = get_orchestrator()
            
            start_time = time.time()
            result = orchestrator.process_query(question)
            processing_time = time.time() - start_time
            
            # STANDARDIZED response format
            return Response({
                'success': result.get('success', True),
                'message': result.get('answer', ''),  # Standardized field name
                'sources': result.get('sources', []) if include_sources else [],
                'tools_used': [tool['tool'] for tool in result.get('tool_results', [])],  # Simplified tool list
                'confidence': result.get('confidence', 0.8),
                'session_id': session_id,
                'processing_time': round(processing_time, 2),
                'timestamp': int(time.time()),
                'metadata': {
                    'sources_found': len(result.get('sources', [])),
                    'response_type': 'orchestrated',
                    'model_used': 'gpt-4o',
                    'tools_executed': len(result.get('tool_results', []))
                }
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Orchestrated chat query failed: {str(e)}")
            return Response({
                'success': False,
                'message': 'I encountered an error processing your request. Please try again.',
                'error_code': 'PROCESSING_ERROR',
                'session_id': request.data.get('session_id', ''),
                'timestamp': int(time.time()),
                'error_details': str(e)  # Keep for debugging
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ChatUploadView(APIView):
    """
    Handle file uploads with full processing pipeline including embeddings
    Enhanced with comprehensive validation and security checks
    """
    
    def post(self, request):
        try:
            # Validation 1: File presence
            if 'file' not in request.FILES:
                return Response({
                    'success': False,
                    'message': 'No file provided',
                    'error_code': 'NO_FILE',
                    'timestamp': int(time.time())
                }, status=status.HTTP_400_BAD_REQUEST)
            
            file = request.FILES['file']
            
            # Validation 2: File size (100MB limit)
            MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
            if file.size > MAX_FILE_SIZE:
                file_size_mb = file.size // (1024 * 1024)
                return Response({
                    'success': False,
                    'message': f'File too large. Maximum size is 100MB, your file is {file_size_mb}MB.',
                    'error_code': 'FILE_TOO_LARGE',
                    'timestamp': int(time.time())
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Validation 3: File type
            file_extension = file.name.split('.')[-1].lower() if '.' in file.name else ''
            allowed_types = ['pdf', 'docx', 'xlsx']
            
            if file_extension not in allowed_types:
                return Response({
                    'success': False,
                    'message': f'File type ".{file_extension}" not supported. Allowed types: {", ".join(allowed_types)}',
                    'error_code': 'UNSUPPORTED_TYPE',
                    'timestamp': int(time.time())
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Validation 4: Filename sanitization (security)
            filename = request.data.get('filename', file.name.split('.')[0])
            # Remove potentially dangerous characters
            filename = re.sub(r'[^\w\s-]', '', filename).strip()
            if not filename:
                filename = f"document_{int(time.time())}"
            
            file_type = request.data.get('type', file_extension)
            
            # Create FileUpload record
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
                    'message': f'File validation failed: {serializer.errors}',
                    'error_code': 'VALIDATION_FAILED',
                    'timestamp': int(time.time())
                }, status=status.HTTP_400_BAD_REQUEST)
            
            file_upload = serializer.save()
            
            # Process with enhanced pipeline
            from apps.ingestion.enhanced_upload_pipeline import EnhancedUploadPipeline
            
            pipeline = EnhancedUploadPipeline()
            result = pipeline.process_uploaded_file(file_upload)
            
            if result['success']:
                return Response({
                    'success': True,
                    'message': 'Document uploaded and processed successfully',
                    'data': {
                        'document_id': result['document_id'],
                        'filename': filename,
                        'file_type': file_type,
                        'status': 'completed',
                        'processing_results': {
                            'text_segments': result['processing_results'].get('text_segments_created', 0),
                            'embeddings': result['processing_results'].get('embeddings_created', 0),
                            'tables': result['processing_results'].get('tables_extracted', 0),
                            'key_values': result['processing_results'].get('key_values_extracted', 0)
                        },
                        'ready_for_queries': True
                    },
                    'timestamp': int(time.time())
                })
            else:
                return Response({
                    'success': False,
                    'message': 'Document uploaded but processing failed',
                    'error_code': 'PROCESSING_FAILED',
                    'data': {
                        'document_id': result.get('document_id'),
                        'filename': filename,
                        'errors': result.get('errors', [])
                    },
                    'timestamp': int(time.time())
                }, status=status.HTTP_202_ACCEPTED)
                
        except Exception as e:
            logger.error(f"Upload failed: {str(e)}")
            return Response({
                'success': False,
                'message': 'Upload failed due to server error',
                'error_code': 'UPLOAD_ERROR',
                'error_details': str(e),
                'timestamp': int(time.time())
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ChatDocumentListView(APIView):
    """
    Enhanced document list with system status and health monitoring
    """
    
    def get(self, request):
        """
        Get list of documents with enhanced metadata and system status
        """
        try:
            from apps.ingestion.models import Document, DocumentTextSegment
            
            # Get pagination parameters
            limit = min(int(request.GET.get('limit', 20)), 100)  # Max 100 documents
            offset = int(request.GET.get('offset', 0))
            
            # Get documents with embedding status
            documents = Document.objects.all().order_by('-uploaded_at')[offset:offset+limit]
            
            document_list = []
            for doc in documents:
                # Calculate embedding coverage
                segments_count = doc.text_segments.count() if hasattr(doc, 'text_segments') else 0
                embedded_count = doc.text_segments.filter(embedding__isnull=False).count() if hasattr(doc, 'text_segments') else 0
                coverage = round((embedded_count / max(segments_count, 1)) * 100, 1)
                
                document_list.append({
                    'id': str(doc.id),
                    'filename': doc.filename,
                    'file_type': doc.file_type,
                    'uploaded_at': doc.uploaded_at.isoformat(),
                    'processing_status': doc.processing_status,
                    'file_size': doc.file_size or 0,
                    'segments': {
                        'total': segments_count,
                        'embedded': embedded_count,
                        'coverage': coverage
                    },
                    'ready_for_queries': embedded_count > 0
                })
            
            # System health check
            system_health = self._get_system_health()
            
            # Statistics
            total_documents = Document.objects.count()
            total_embeddings = DocumentTextSegment.objects.filter(embedding__isnull=False).count()
            
            return Response({
                'success': True,
                'system': system_health,
                'documents': document_list,
                'pagination': {
                    'total': total_documents,
                    'limit': limit,
                    'offset': offset,
                    'has_next': offset + limit < total_documents
                },
                'statistics': {
                    'total_documents': total_documents,
                    'total_embeddings': total_embeddings,
                    'documents_ready': len([d for d in document_list if d['ready_for_queries']]),
                    'system_ready': total_embeddings > 0
                },
                'timestamp': int(time.time())
            })
            
        except Exception as e:
            logger.error(f"Document list failed: {str(e)}")
            return Response({
                'success': False,
                'message': f'Failed to retrieve documents: {str(e)}',
                'error_code': 'DOCUMENT_LIST_ERROR',
                'timestamp': int(time.time())
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def _get_system_health(self):
        """Quick system health check"""
        health = {'status': 'healthy', 'components': {}}
        
        try:
            # Check RAG engine
            rag_engine = get_rag_engine()
            rag_health = rag_engine.health_check()
            health['components']['rag'] = {
                'status': rag_health.get('overall_status', 'unknown'),
                'embedding_count': rag_health.get('embedding_count', 0)
            }
        except Exception as e:
            health['components']['rag'] = {'status': 'unhealthy', 'error': str(e)}
        
        try:
            # Check orchestrator
            orchestrator = get_orchestrator()
            health['components']['orchestrator'] = {'status': 'healthy', 'model': 'gpt-4o'}
        except Exception as e:
            health['components']['orchestrator'] = {'status': 'unhealthy', 'error': str(e)}
        
        # Overall status
        component_statuses = [comp.get('status', 'unhealthy') for comp in health['components'].values()]
        if all(status == 'healthy' for status in component_statuses):
            health['status'] = 'healthy'
        elif any(status == 'healthy' for status in component_statuses):
            health['status'] = 'degraded'
        else:
            health['status'] = 'unhealthy'
        
        return health