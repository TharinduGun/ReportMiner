"""
Enhanced URLs configuration for ReportMiner ingestion app
"""

from django.urls import path
from .views import (
    # Enhanced views
    EnhancedFileUploadView,
    DocumentSearchView,
    DocumentDetailView,
    DocumentListView,
    SystemStatsView,
    
    # Function-based views
    process_legacy_uploads_view,
    document_content_view,
    document_tables_view,
    reprocess_document_view,
    
    # Legacy view (for backward compatibility)
    FileUploadView
)

app_name = 'ingestion'

urlpatterns = [
    # === ENHANCED UPLOAD PIPELINE ===
    
    # Primary upload endpoint (enhanced)
    path('upload/enhanced/', EnhancedFileUploadView.as_view(), name='enhanced-file-upload'),
    
    # Legacy upload endpoint (backward compatibility)
    path('upload/', FileUploadView.as_view(), name='file-upload'),
    
    # === DOCUMENT MANAGEMENT ===
    
    # List all documents with filtering
    path('documents/', DocumentListView.as_view(), name='document-list'),
    
    # Get detailed document information
    path('documents/<uuid:document_id>/', DocumentDetailView.as_view(), name='document-detail'),
    
    # Get document text content and segments
    path('documents/<uuid:document_id>/content/', document_content_view, name='document-content'),
    
    # Get document tables and structured data
    path('documents/<uuid:document_id>/tables/', document_tables_view, name='document-tables'),
    
    # Reprocess a specific document
    path('documents/<uuid:document_id>/reprocess/', reprocess_document_view, name='document-reprocess'),
    
    # === SEARCH AND QUERY ===
    
    # Search documents by text query
    path('search/', DocumentSearchView.as_view(), name='document-search'),
    
    # === SYSTEM MANAGEMENT ===
    
    # Get system processing statistics
    path('stats/', SystemStatsView.as_view(), name='system-stats'),
    
    # Process legacy uploads
    path('process-legacy/', process_legacy_uploads_view, name='process-legacy-uploads'),
]

"""
API Endpoint Documentation:

=== UPLOAD ENDPOINTS ===

POST /api/ingestion/upload/enhanced/
- Enhanced file upload with full processing pipeline
- Body: multipart/form-data with 'file', 'filename', 'type'
- Returns: Document ID, processing stats, text preview

POST /api/ingestion/upload/
- Legacy upload endpoint (backward compatibility)
- Body: multipart/form-data with 'file', 'filename', 'type'
- Returns: Basic text extraction only

=== DOCUMENT ENDPOINTS ===

GET /api/ingestion/documents/
- List all documents with optional filtering
- Query params: file_type, status, document_type, limit, offset
- Returns: Paginated list of documents

GET /api/ingestion/documents/{document_id}/
- Get detailed document information
- Returns: Document metadata and processing statistics

GET /api/ingestion/documents/{document_id}/content/
- Get document text segments and key-value pairs
- Returns: All extracted content segments

GET /api/ingestion/documents/{document_id}/tables/
- Get document tables and structured data
- Returns: Table structures and cell data

POST /api/ingestion/documents/{document_id}/reprocess/
- Reprocess a document through the pipeline
- Returns: New processing results

=== SEARCH ENDPOINTS ===

GET /api/ingestion/search/?q={query}
- Search documents using full-text search
- Query params: q (required), type, limit
- Returns: Ranked search results

=== SYSTEM ENDPOINTS ===

GET /api/ingestion/stats/
- Get system processing statistics
- Returns: Document counts, processing stats

POST /api/ingestion/process-legacy/
- Process existing FileUpload records
- Returns: Batch processing results

=== EXAMPLE USAGE ===

# Upload and process a document
curl -X POST http://localhost:8000/api/ingestion/upload/enhanced/ \
  -F "file=@document.pdf" \
  -F "filename=document.pdf" \
  -F "type=pdf"

# Search for documents
curl "http://localhost:8000/api/ingestion/search/?q=financial%20report&limit=5"

# Get document details
curl "http://localhost:8000/api/ingestion/documents/{document-id}/"

# List all PDF documents
curl "http://localhost:8000/api/ingestion/documents/?file_type=pdf&limit=10"

# Get system statistics
curl "http://localhost:8000/api/ingestion/stats/"
"""