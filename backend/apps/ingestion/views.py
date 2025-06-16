"""
Enhanced Views for ReportMiner with integrated upload pipeline
"""

from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from .serializers import FileUploadSerializer
from .models import Document, DocumentTextSegment, DocumentKeyValue, FileUpload
from .enhanced_upload_pipeline import EnhancedUploadPipeline, DocumentSearchPipeline, process_legacy_uploads, get_processing_statistics
from .utils import preview_text


class EnhancedFileUploadView(APIView):
    """
    Enhanced file upload view that uses the new processing pipeline
    """
    
    def post(self, request):
        """Handle file upload with enhanced processing"""
        serializer = FileUploadSerializer(data=request.data)
        
        if serializer.is_valid():
            # Save the file upload (legacy compatibility)
            file_upload = serializer.save()
            
            try:
                # Process with enhanced pipeline
                pipeline = EnhancedUploadPipeline()
                result = pipeline.process_uploaded_file(file_upload)
                
                if result['success']:
                    return Response({
                        "success": True,
                        "message": "File uploaded and processed successfully!",
                        "data": {
                            "filename": file_upload.filename,
                            "type": file_upload.type,
                            "document_id": result['document_id'],
                            "text_preview": result['text_preview'],
                            "processing_stats": result['processing_results'],
                            "upload_id": file_upload.id
                        }
                    }, status=status.HTTP_201_CREATED)
                else:
                    return Response({
                        "success": False,
                        "message": "File uploaded but processing failed",
                        "errors": result['errors'],
                        "data": {
                            "filename": file_upload.filename,
                            "document_id": result.get('document_id'),
                            "upload_id": file_upload.id
                        }
                    }, status=status.HTTP_202_ACCEPTED)
                    
            except Exception as e:
                return Response({
                    "success": False,
                    "message": "Processing pipeline error",
                    "error": str(e),
                    "data": {
                        "filename": file_upload.filename,
                        "upload_id": file_upload.id
                    }
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        return Response({
            "success": False,
            "message": "Invalid file upload",
            "errors": serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)


class DocumentSearchView(APIView):
    """
    API view for searching processed documents
    """
    
    def get(self, request):
        """Search documents by text query"""
        query = request.GET.get('q', '').strip()
        document_type = request.GET.get('type')
        limit = min(int(request.GET.get('limit', 10)), 50)  # Max 50 results
        
        if not query:
            return Response({
                "error": "Query parameter 'q' is required"
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            results = DocumentSearchPipeline.search_documents(
                query=query,
                document_type=document_type,
                limit=limit
            )
            
            return Response({
                "success": True,
                "search": results
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({
                "success": False,
                "error": f"Search failed: {str(e)}"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class DocumentDetailView(APIView):
    """
    API view for getting detailed document information
    """
    
    def get(self, request, document_id):
        """Get detailed information about a specific document"""
        try:
            summary = DocumentSearchPipeline.get_document_summary(document_id)
            
            if 'error' in summary:
                return Response({
                    "success": False,
                    "error": summary['error']
                }, status=status.HTTP_404_NOT_FOUND)
            
            return Response({
                "success": True,
                "document": summary
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({
                "success": False,
                "error": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class DocumentListView(APIView):
    """
    API view for listing all processed documents
    """
    
    def get(self, request):
        """Get list of all processed documents with filters"""
        try:
            # Get query parameters
            file_type = request.GET.get('file_type')
            processing_status = request.GET.get('status')
            document_type = request.GET.get('document_type')
            limit = min(int(request.GET.get('limit', 20)), 100)
            offset = int(request.GET.get('offset', 0))
            
            # Build query
            queryset = Document.objects.all()
            
            if file_type:
                queryset = queryset.filter(file_type=file_type)
            if processing_status:
                queryset = queryset.filter(processing_status=processing_status)
            if document_type:
                queryset = queryset.filter(document_type=document_type)
            
            # Get total count
            total_count = queryset.count()
            
            # Apply pagination
            documents = queryset.order_by('-uploaded_at')[offset:offset + limit]
            
            # Format response
            document_list = []
            for doc in documents:
                document_list.append({
                    'id': str(doc.id),
                    'filename': doc.filename,
                    'file_type': doc.file_type,
                    'document_type': doc.document_type,
                    'processing_status': doc.processing_status,
                    'uploaded_at': doc.uploaded_at.isoformat(),
                    'file_size': doc.file_size,
                    'segments_count': doc.text_segments.count(),
                    'tables_count': doc.tables.count(),
                    'key_values_count': doc.key_values.count()
                })
            
            return Response({
                "success": True,
                "data": {
                    "documents": document_list,
                    "pagination": {
                        "total": total_count,
                        "limit": limit,
                        "offset": offset,
                        "has_next": offset + limit < total_count
                    }
                }
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({
                "success": False,
                "error": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SystemStatsView(APIView):
    """
    API view for getting system processing statistics
    """
    
    def get(self, request):
        """Get overall system statistics"""
        try:
            stats = get_processing_statistics()
            return Response({
                "success": True,
                "statistics": stats
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({
                "success": False,
                "error": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# Function-based views for specific operations
@api_view(['POST'])
def process_legacy_uploads_view(request):
    """
    Process any existing FileUpload records that haven't been processed
    """
    try:
        results = process_legacy_uploads()
        return Response({
            "success": True,
            "message": "Legacy uploads processing completed",
            "results": results
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({
            "success": False,
            "error": str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def document_content_view(request, document_id):
    """
    Get the content segments of a specific document
    """
    try:
        document = Document.objects.get(id=document_id)
        
        # Get text segments
        segments = DocumentTextSegment.objects.filter(
            document=document
        ).order_by('sequence_number').values(
            'id', 'sequence_number', 'content', 'segment_type', 
            'section_title', 'word_count'
        )
        
        # Get key-value pairs
        key_values = DocumentKeyValue.objects.filter(
            document=document
        ).values(
            'key_name', 'value_text', 'data_type', 'key_category'
        )
        
        return Response({
            "success": True,
            "document": {
                "id": str(document.id),
                "filename": document.filename,
                "processing_status": document.processing_status
            },
            "content": {
                "text_segments": list(segments),
                "key_values": list(key_values),
                "total_segments": len(segments),
                "total_key_values": len(key_values)
            }
        }, status=status.HTTP_200_OK)
        
    except Document.DoesNotExist:
        return Response({
            "success": False,
            "error": "Document not found"
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({
            "success": False,
            "error": str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def document_tables_view(request, document_id):
    """
    Get the tables and structured data of a specific document
    """
    try:
        document = Document.objects.get(id=document_id)
        
        tables_data = []
        for table in document.tables.all():
            # Get structured data for this table
            cells = table.cells.all().values(
                'row_number', 'column_number', 'column_name', 
                'cell_value', 'data_type', 'numeric_value'
            )
            
            tables_data.append({
                'id': str(table.id),
                'table_name': table.table_name,
                'row_count': table.row_count,
                'column_count': table.column_count,
                'has_header': table.has_header,
                'cells': list(cells)
            })
        
        return Response({
            "success": True,
            "document": {
                "id": str(document.id),
                "filename": document.filename
            },
            "tables": tables_data
        }, status=status.HTTP_200_OK)
        
    except Document.DoesNotExist:
        return Response({
            "success": False,
            "error": "Document not found"
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({
            "success": False,
            "error": str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def reprocess_document_view(request, document_id):
    """
    Reprocess a specific document through the pipeline
    """
    try:
        document = Document.objects.get(id=document_id)
        
        # Check if we have the original FileUpload
        file_upload = None
        try:
            metadata = eval(document.metadata) if document.metadata != '{}' else {}
            legacy_id = metadata.get('legacy_upload_id')
            if legacy_id:
                file_upload = FileUpload.objects.get(id=legacy_id)
        except:
            pass
        
        if not file_upload:
            return Response({
                "success": False,
                "error": "Original file upload not found for reprocessing"
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Clear existing processed data
        document.text_segments.all().delete()
        document.tables.all().delete()
        document.key_values.all().delete()
        document.structured_data.all().delete()
        
        # Reprocess
        pipeline = EnhancedUploadPipeline()
        result = pipeline._process_document_content(
            document, 
            pipeline._extract_text(file_upload)
        )
        
        return Response({
            "success": True,
            "message": "Document reprocessed successfully",
            "processing_results": result
        }, status=status.HTTP_200_OK)
        
    except Document.DoesNotExist:
        return Response({
            "success": False,
            "error": "Document not found"
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({
            "success": False,
            "error": str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# Keep the original FileUploadView for backward compatibility
class FileUploadView(APIView):
    """
    Original file upload view (maintained for backward compatibility)
    """
    
    def post(self, request):
        serializer = FileUploadSerializer(data=request.data)
        
        if serializer.is_valid():
            instance = serializer.save()
            
            file_path = instance.file.path
            file_type = instance.type
            
            # Extract text based on file type
            from .extractor import extract_text_from_file
            text = extract_text_from_file(file_path, file_type)
            
            return Response({
                "message": "File uploaded and text extracted successfully!",
                "filename": instance.filename,
                "type": instance.type,
                "text_preview": preview_text(text),
                "upload_id": instance.id
            }, status=status.HTTP_201_CREATED)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)