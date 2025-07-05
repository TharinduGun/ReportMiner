"""
Vector Processor for ReportMiner
Handles embedding generation and semantic search using OpenAI and pgvector
"""

import openai
import numpy as np
from typing import List, Dict, Any, Optional
from django.conf import settings
from django.db import connection
from .models import DocumentTextSegment, Document


class VectorProcessor:
    """Handle vector embeddings for semantic search"""
    
    def __init__(self):
        """Initialize with OpenAI client"""
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = "text-embedding-ada-002"
        self.dimensions = 1536
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for text using OpenAI
        
        Args:
            text: Text content to embed
            
        Returns:
            List of floats representing the embedding, or None if error
        """
        try:
            # Clean and truncate text if too long (OpenAI has token limits)
            text = text.strip()
            if len(text) > 8000:  # Rough token limit safety
                text = text[:8000]
            
            if not text:
                return None
            
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            
            embedding = response.data[0].embedding
            
            # Verify embedding dimensions
            if len(embedding) != self.dimensions:
                print(f"Warning: Expected {self.dimensions} dimensions, got {len(embedding)}")
            
            return embedding
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def generate_embeddings_for_document(self, document_id: str) -> Dict[str, Any]:
        """
        Generate embeddings for all text segments of a document
        
        Args:
            document_id: UUID of the document
            
        Returns:
            Dictionary with processing results
        """
        results = {
            'success': False,
            'processed_segments': 0,
            'failed_segments': 0,
            'errors': []
        }
        
        try:
            # Get all text segments for this document that don't have embeddings
            segments = DocumentTextSegment.objects.filter(
                document_id=document_id,
                embedding__isnull=True
            ).exclude(content='')
            
            print(f"Processing {segments.count()} segments for document {document_id}")
            
            for segment in segments:
                try:
                    # Generate embedding
                    embedding = self.generate_embedding(segment.content)
                    
                    if embedding:
                        # Save embedding to database
                        segment.embedding = embedding
                        segment.embedding_model = self.model
                        segment.save()
                        
                        results['processed_segments'] += 1
                        print(f"✓ Processed segment {segment.sequence_number}")
                    else:
                        results['failed_segments'] += 1
                        results['errors'].append(f"Failed to generate embedding for segment {segment.sequence_number}")
                        
                except Exception as e:
                    results['failed_segments'] += 1
                    error_msg = f"Error processing segment {segment.sequence_number}: {str(e)}"
                    results['errors'].append(error_msg)
                    print(f"✗ {error_msg}")
            
            # Update document status
            document = Document.objects.get(id=document_id)
            total_segments = document.text_segments.count()
            segments_with_embeddings = document.text_segments.filter(embedding__isnull=False).count()
            
            if segments_with_embeddings == total_segments:
                # All segments have embeddings
                document.metadata = document.metadata.replace(
                    '"embedding_status": "pending"',
                    '"embedding_status": "completed"'
                )
            else:
                # Partial embeddings
                document.metadata = document.metadata.replace(
                    '"embedding_status": "pending"',
                    '"embedding_status": "partial"'
                )
            
            document.save()
            results['success'] = True
            
        except Exception as e:
            results['errors'].append(f"Document processing error: {str(e)}")
            print(f"Error processing document {document_id}: {e}")
        
        return results
    
    def semantic_search(self, query: str, limit: int = 10, document_type: Optional[str] = None) -> List[Dict]:
        """
        Perform semantic search using vector similarity with Django ORM
        """
        try:
            from pgvector.django import CosineDistance
            
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            
            if not query_embedding:
                return []
            
            # Use Django ORM with pgvector - this handles type conversion automatically
            queryset = DocumentTextSegment.objects.filter(
                embedding__isnull=False
            ).select_related('document')
            
            # Add document type filter if specified
            if document_type:
                queryset = queryset.filter(document__document_type=document_type)
            
            # Add distance annotation and order by similarity
            queryset = queryset.annotate(
                distance=CosineDistance('embedding', query_embedding)
            ).order_by('distance')[:limit]
            
            # Format results
            formatted_results = []
            for segment in queryset:
                distance = float(segment.distance)
                similarity = 1 - distance
                
                formatted_results.append({
                    'segment_id': str(segment.id),
                    'content': segment.content,
                    'sequence_number': segment.sequence_number,
                    'document_id': str(segment.document.id),
                    'filename': segment.document.filename,
                    'document_type': segment.document.document_type,
                    'distance': distance,
                    'similarity_score': similarity
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in semantic search: {e}")
            import traceback
            traceback.print_exc()
            return []   
        
        
    def hybrid_search(self, query: str, limit: int = 10, keyword_weight: float = 0.3, semantic_weight: float = 0.7) -> List[Dict]:
        """
        Combine keyword and semantic search for better results
        
        Args:
            query: Search query
            limit: Maximum results
            keyword_weight: Weight for keyword search (0-1)
            semantic_weight: Weight for semantic search (0-1)
            
        Returns:
            Combined search results
        """
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            
            if not query_embedding:
                # Fall back to keyword search only
                return self._keyword_search(query, limit)
        
            # Convert to PostgreSQL vector format
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Hybrid search SQL
            hybrid_query = """
                SELECT 
                    ts.id as segment_id,
                    ts.content,
                    ts.sequence_number,
                    d.id as document_id,
                    d.filename,
                    d.document_type,
                    ts.embedding <-> %s as vector_distance,
                    ts_rank(to_tsvector('english', ts.content), plainto_tsquery('english', %s)) as keyword_rank,
                    (
                        %s * ts_rank(to_tsvector('english', ts.content), plainto_tsquery('english', %s)) +
                        %s * (1 - (ts.embedding <-> %s))
                    ) as combined_score
                FROM document_text_segments ts
                JOIN documents d ON ts.document_id = d.id
                WHERE ts.embedding IS NOT NULL
                  AND (
                    to_tsvector('english', ts.content) @@ plainto_tsquery('english', %s)
                    OR (ts.embedding <-> %s) < 0.5
                  )
                ORDER BY combined_score DESC
                LIMIT %s
            """
            
            params = [
                embedding_str, query, 
                keyword_weight, query,
                semantic_weight, embedding_str,
                query, embedding_str,
                limit
            ]
            
            with connection.cursor() as cursor:
                cursor.execute(hybrid_query, params)
                results = cursor.fetchall()
            
            # Format results
            formatted_results = []
            for row in results:
                formatted_results.append({
                    'segment_id': str(row[0]),
                    'content': row[1],
                    'sequence_number': row[2],
                    'document_id': str(row[3]),
                    'filename': row[4],
                    'document_type': row[5],
                    'vector_distance': float(row[6]),
                    'keyword_rank': float(row[7]),
                    'combined_score': float(row[8]),
                    'similarity_score': 1 - float(row[6])
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in hybrid search: {e}")
            return []
    
    def _keyword_search(self, query: str, limit: int) -> List[Dict]:
        """Fallback keyword search when embeddings not available"""
        try:
            keyword_query = """
                SELECT 
                    ts.id as segment_id,
                    ts.content,
                    ts.sequence_number,
                    d.id as document_id,
                    d.filename,
                    d.document_type,
                    ts_rank(to_tsvector('english', ts.content), plainto_tsquery('english', %s)) as keyword_rank
                FROM document_text_segments ts
                JOIN documents d ON ts.document_id = d.id
                WHERE to_tsvector('english', ts.content) @@ plainto_tsquery('english', %s)
                ORDER BY keyword_rank DESC
                LIMIT %s
            """
            
            with connection.cursor() as cursor:
                cursor.execute(keyword_query, [query, query, limit])
                results = cursor.fetchall()
            
            formatted_results = []
            for row in results:
                formatted_results.append({
                    'segment_id': str(row[0]),
                    'content': row[1],
                    'sequence_number': row[2],
                    'document_id': str(row[3]),
                    'filename': row[4],
                    'document_type': row[5],
                    'keyword_rank': float(row[6]),
                    'search_type': 'keyword_only'
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in keyword search: {e}")
            return []


# Utility functions for easy access
def generate_document_embeddings(document_id: str) -> Dict[str, Any]:
    """
    Convenience function to generate embeddings for a document
    """
    processor = VectorProcessor()
    return processor.generate_embeddings_for_document(document_id)


def search_documents(query: str, search_type: str = 'semantic', limit: int = 10) -> List[Dict]:
    """
    Convenience function for document search
    
    Args:
        query: Search query
        search_type: 'semantic', 'keyword', or 'hybrid'
        limit: Maximum results
    """
    processor = VectorProcessor()
    
    if search_type == 'semantic':
        return processor.semantic_search(query, limit)
    elif search_type == 'hybrid':
        return processor.hybrid_search(query, limit)
    else:  # keyword
        return processor._keyword_search(query, limit)
    


