import os
import json
import mimetypes
from typing import Dict, Any, Optional
from django.core.files.storage import default_storage
from django.utils import timezone
from django.db import transaction
from django.db.models import Count, Q

# Import models with proper relative imports
from .models import Document, FileUpload, DocumentTextSegment, DocumentTable, DocumentStructuredData, DocumentKeyValue
from .extractor import extract_text_from_file
from .text_processor import TextProcessor
from .utils import get_file_extension, preview_text


class EnhancedUploadPipeline:
    """
    Enhanced upload pipeline that handles the complete document processing workflow
    """
    
    def __init__(self):
        self.text_processor = TextProcessor()
    
    def process_uploaded_file(self, file_upload: FileUpload) -> Dict[str, Any]:
        """
        Main processing function that coordinates the entire upload pipeline
        
        Args:
            file_upload: FileUpload instance from the legacy upload
            
        Returns:
            Dict containing processing results and document information
        """
        result = {
            'success': False,
            'document_id': None,
            'processing_results': {},
            'errors': [],
            'text_preview': ''
        }
        
        try:
            # Step 1: Create main Document record
            document = self._create_document_record(file_upload)
            result['document_id'] = str(document.id)
            
            # Step 2: Extract text from the uploaded file
            raw_text = self._extract_text(file_upload)
            result['text_preview'] = preview_text(raw_text)
            
            # Step 3: Process the extracted text
            processing_results = self._process_document_content(document, raw_text)
            result['processing_results'] = processing_results
            
            # Step 4: Update final status
            if processing_results.get('processing_status') == 'completed':
                result['success'] = True
            else:
                result['errors'].extend(processing_results.get('errors', []))
                
        except Exception as e:
            result['errors'].append(f"Pipeline error: {str(e)}")
            if result['document_id']:
                # Update document status to failed
                try:
                    document = Document.objects.get(id=result['document_id'])
                    document.processing_status = 'failed'
                    document.processing_error = str(e)
                    document.save()
                except:
                    pass
        
        return result
    
    def _create_document_record(self, file_upload: FileUpload) -> Document:
        """
        Create a Document record from FileUpload data
        
        Args:
            file_upload: Legacy FileUpload instance
            
        Returns:
            Created Document instance
        """
        # Get file information
        file_path = file_upload.file.path if file_upload.file else None
        file_size = None
        mime_type = None
        
        if file_path and os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            mime_type, _ = mimetypes.guess_type(file_path)
        
        # Create metadata
        metadata = {
            'legacy_upload_id': file_upload.id,
            'original_upload_time': file_upload.uploaded_at.isoformat(),
            'extraction_method': 'automated_pipeline',
            'file_source': 'web_upload'
        }
        
        # Create Document record
        document = Document.objects.create(
            filename=file_upload.filename,
            original_filename=file_upload.filename,
            file_path=file_path,
            file_type=file_upload.type,
            file_size=file_size,
            mime_type=mime_type,
            processing_status='pending',
            metadata=json.dumps(metadata),
            document_type=self._classify_document_type(file_upload.type, file_upload.filename),
            uploaded_at=file_upload.uploaded_at
        )
        
        return document
    
    def _classify_document_type(self, file_type: str, filename: str) -> str:
        """Classify document type based on file type and name"""
        filename_lower = filename.lower()
        
        # Financial documents
        if any(term in filename_lower for term in ['financial', 'report', 'statement', 'budget']):
            return 'financial_report'
        
        # Research documents
        if any(term in filename_lower for term in ['research', 'study', 'analysis', 'whitepaper']):
            return 'research_document'
        
        # Contracts and legal
        if any(term in filename_lower for term in ['contract', 'agreement', 'legal', 'terms']):
            return 'legal_document'
        
        # General classification by file type
        if file_type == 'pdf':
            return 'pdf_document'
        elif file_type == 'docx':
            return 'word_document'
        elif file_type == 'xlsx':
            return 'spreadsheet'
        
        return 'general_document'
    
    def _extract_text(self, file_upload: FileUpload) -> str:
        """
        Extract text from the uploaded file using existing extractor
        
        Args:
            file_upload: FileUpload instance
            
        Returns:
            Extracted text content
        """
        file_path = file_upload.file.path
        file_type = file_upload.type
        
        # Use existing extractor
        raw_text = extract_text_from_file(file_path, file_type)
        
        # Check for extraction errors
        if raw_text.startswith('[') and 'Error]' in raw_text:
            raise Exception(f"Text extraction failed: {raw_text}")
        
        return raw_text
    
    def _process_document_content(self, document: Document, raw_text: str) -> Dict[str, Any]:
        """
        Process the extracted text and store structured data
        
        Args:
            document: Document instance
            raw_text: Extracted text content
            
        Returns:
            Processing results
        """
        return self.text_processor.process_document_text(document, raw_text)


class BatchUploadPipeline:
    """
    Enhanced pipeline for processing multiple files in batch
    """
    
    def __init__(self):
        self.upload_pipeline = EnhancedUploadPipeline()
    
    def process_batch(self, file_uploads: list) -> Dict[str, Any]:
        """
        Process multiple FileUpload instances in batch
        
        Args:
            file_uploads: List of FileUpload instances
            
        Returns:
            Batch processing results
        """
        batch_results = {
            'total_files': len(file_uploads),
            'successful': 0,
            'failed': 0,
            'results': [],
            'errors': []
        }
        
        for file_upload in file_uploads:
            try:
                result = self.upload_pipeline.process_uploaded_file(file_upload)
                batch_results['results'].append({
                    'file_upload_id': file_upload.id,
                    'filename': file_upload.filename,
                    'document_id': result.get('document_id'),
                    'success': result['success'],
                    'errors': result.get('errors', [])
                })
                
                if result['success']:
                    batch_results['successful'] += 1
                else:
                    batch_results['failed'] += 1
                    
            except Exception as e:
                batch_results['failed'] += 1
                batch_results['errors'].append(f"Error processing {file_upload.filename}: {str(e)}")
                batch_results['results'].append({
                    'file_upload_id': file_upload.id,
                    'filename': file_upload.filename,
                    'success': False,
                    'errors': [str(e)]
                })
        
        return batch_results


class DocumentSearchPipeline:
    """
    Pipeline for searching and retrieving processed documents
    """
    
    @staticmethod
    def search_documents(query: str, document_type: Optional[str] = None, 
                        limit: int = 10) -> Dict[str, Any]:
        """
        Search documents using PostgreSQL full-text search
        
        Args:
            query: Search query string
            document_type: Optional document type filter
            limit: Maximum number of results
            
        Returns:
            Search results with document and segment information
        """
        from django.db import connection
        
        # Use the built-in search function
        with connection.cursor() as cursor:
            if document_type:
                cursor.execute("""
                    SELECT d.id, d.filename, ts.id, ts.content, 
                           ts_rank(to_tsvector('english', ts.content), plainto_tsquery('english', %s)) as rank_score
                    FROM documents d
                    JOIN document_text_segments ts ON d.id = ts.document_id
                    WHERE to_tsvector('english', ts.content) @@ plainto_tsquery('english', %s)
                      AND d.document_type = %s
                    ORDER BY rank_score DESC
                    LIMIT %s
                """, [query, query, document_type, limit])
            else:
                cursor.execute("""
                    SELECT d.id, d.filename, ts.id, ts.content,
                           ts_rank(to_tsvector('english', ts.content), plainto_tsquery('english', %s)) as rank_score
                    FROM documents d
                    JOIN document_text_segments ts ON d.id = ts.document_id
                    WHERE to_tsvector('english', ts.content) @@ plainto_tsquery('english', %s)
                    ORDER BY rank_score DESC
                    LIMIT %s
                """, [query, query, limit])
            
            results = cursor.fetchall()
        
        # Format results
        formatted_results = []
        for row in results:
            formatted_results.append({
                'document_id': str(row[0]),
                'filename': row[1],
                'segment_id': str(row[2]),
                'content': row[3],
                'relevance_score': float(row[4])
            })
        
        return {
            'query': query,
            'total_results': len(formatted_results),
            'results': formatted_results
        }
    
    @staticmethod
    def get_document_summary(document_id: str) -> Dict[str, Any]:
        """
        Get comprehensive summary of a processed document
        
        Args:
            document_id: UUID of the document
            
        Returns:
            Document summary with all extracted data
        """
        try:
            document = Document.objects.get(id=document_id)
            
            # Get related data
            segments_count = document.text_segments.count()
            tables_count = document.tables.count()
            key_values_count = document.key_values.count()
            structured_data_count = document.structured_data.count()
            
            # Get sample content
            sample_segments = list(document.text_segments.values(
                'sequence_number', 'content', 'segment_type'
            )[:3])
            
            sample_key_values = list(document.key_values.values(
                'key_name', 'value_text', 'data_type'
            )[:5])
            
            summary = {
                'document': {
                    'id': str(document.id),
                    'filename': document.filename,
                    'file_type': document.file_type,
                    'processing_status': document.processing_status,
                    'uploaded_at': document.uploaded_at.isoformat(),
                    'document_type': document.document_type
                },
                'statistics': {
                    'text_segments': segments_count,
                    'tables': tables_count,
                    'key_values': key_values_count,
                    'structured_data_cells': structured_data_count
                },
                'sample_content': {
                    'text_segments': sample_segments,
                    'key_values': sample_key_values
                }
            }
            
            return summary
            
        except Document.DoesNotExist:
            return {'error': 'Document not found'}
        except Exception as e:
            return {'error': str(e)}


# Helper functions for the pipeline
def process_legacy_uploads():
    """
    Process any existing FileUpload records that haven't been processed yet
    """
    # Get FileUploads that don't have corresponding Documents
    processed_filenames = set(Document.objects.values_list('filename', flat=True))
    unprocessed_uploads = FileUpload.objects.exclude(filename__in=processed_filenames)
    
    if unprocessed_uploads.exists():
        pipeline = BatchUploadPipeline()
        results = pipeline.process_batch(list(unprocessed_uploads))
        return results
    
    return {'message': 'No unprocessed uploads found'}


def get_processing_statistics() -> Dict[str, Any]:
    """
    Get overall processing statistics for the system
    """
    stats = {
        'documents': {
            'total': Document.objects.count(),
            'by_status': dict(Document.objects.values('processing_status').annotate(count=Count('id')).values_list('processing_status', 'count')),
            'by_type': dict(Document.objects.values('file_type').annotate(count=Count('id')).values_list('file_type', 'count'))
        },
        'content': {
            'total_text_segments': DocumentTextSegment.objects.count(),
            'total_tables': DocumentTable.objects.count(),
            'total_key_values': DocumentKeyValue.objects.count(),
            'total_structured_cells': DocumentStructuredData.objects.count()
        }
    }
    
    return stats