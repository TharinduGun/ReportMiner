"""
Test-friendly Vector Processor that works without pgvector
"""
import openai
import numpy as np
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from django.conf import settings
from django.db import connection, transaction
from django.core.cache import cache
from datetime import datetime, timedelta
from .models import DocumentTextSegment, Document
from .embedding_config import EmbeddingConfig

logger = logging.getLogger(__name__)

# Check if pgvector is available
try:
    from pgvector.django import CosineDistance
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False
    # Mock CosineDistance for testing
    class CosineDistance:
        def __init__(self, field, vector):
            self.field = field
            self.vector = vector

class EmbeddingError(Exception):
    """Custom exception for embedding-related errors"""
    pass

class RateLimitError(EmbeddingError):
    """Exception for rate limit errors"""
    pass

class VectorProcessor:
    """Test-friendly vector processor with robust error handling"""
    
    def __init__(self):
        """Initialize with OpenAI client and configuration"""
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = "text-embedding-ada-002"
        self.dimensions = 1536
        self.config = EmbeddingConfig()
        
        # Rate limiting tracking
        self.daily_call_count_key = f"embedding_calls_{datetime.now().strftime('%Y-%m-%d')}"
        
    def check_rate_limits(self) -> bool:
        """Check if we're within daily rate limits"""
        daily_calls = cache.get(self.daily_call_count_key, 0)
        if daily_calls >= self.config.MAX_DAILY_API_CALLS:
            logger.warning(f"Daily embedding limit reached: {daily_calls}")
            return False
        return True
    
    def increment_call_count(self):
        """Increment daily API call counter"""
        current_count = cache.get(self.daily_call_count_key, 0)
        cache.set(self.daily_call_count_key, current_count + 1, 86400)  # 24 hours
        
        # Alert if approaching limit
        if current_count > self.config.COST_ALERT_THRESHOLD:
            logger.warning(f"Embedding usage high: {current_count} calls today")
    
    def should_embed_segment(self, segment: DocumentTextSegment) -> Tuple[bool, str]:
        """Determine if a segment should be embedded"""
        # Check if already has embedding
        if hasattr(segment, 'embedding') and segment.embedding is not None:
            return False, "Already has embedding"
        
        # Check text length
        if not segment.content or len(segment.content.strip()) < self.config.MIN_TEXT_LENGTH:
            return False, f"Text too short ({len(segment.content) if segment.content else 0} chars)"
        
        # Check segment type
        if segment.segment_type in self.config.SKIP_SEGMENT_TYPES:
            return False, f"Skipping segment type: {segment.segment_type}"
        
        # Check rate limits
        if not self.check_rate_limits():
            return False, "Daily rate limit reached"
        
        return True, "OK"
    
    def clean_text(self, text: str) -> str:
        """Clean and prepare text for embedding"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long
        if len(text) > self.config.MAX_TEXT_LENGTH:
            text = text[:self.config.MAX_TEXT_LENGTH]
            logger.info(f"Truncated text to {self.config.MAX_TEXT_LENGTH} characters")
        
        return text.strip()
    
    def generate_embedding_with_retry(self, text: str) -> Optional[List[float]]:
        """Generate embedding with retry logic and error handling"""
        for attempt in range(self.config.MAX_RETRIES):
            try:
                # Clean and prepare text
                cleaned_text = self.clean_text(text)
                if not cleaned_text:
                    return None
                
                # Make API call with timeout
                response = self.client.embeddings.create(
                    input=cleaned_text,
                    model=self.model,
                    timeout=self.config.REQUEST_TIMEOUT
                )
                
                embedding = response.data[0].embedding
                
                # Validate embedding
                if len(embedding) != self.dimensions:
                    raise EmbeddingError(f"Invalid embedding dimensions: {len(embedding)}")
                
                self.increment_call_count()
                logger.debug(f"Successfully generated embedding (attempt {attempt + 1})")
                return embedding
                
            except openai.RateLimitError as e:
                wait_time = self.config.RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Rate limit hit, waiting {wait_time}s (attempt {attempt + 1})")
                if attempt < self.config.MAX_RETRIES - 1:
                    time.sleep(wait_time)
                else:
                    logger.error("Max retries reached for rate limit")
                    
            except openai.APITimeoutError as e:
                logger.warning(f"API timeout (attempt {attempt + 1}): {str(e)}")
                if attempt < self.config.MAX_RETRIES - 1:
                    time.sleep(self.config.RETRY_DELAY)
                else:
                    logger.error("Max retries reached for timeout")
                    
            except openai.APIError as e:
                logger.error(f"OpenAI API error (attempt {attempt + 1}): {str(e)}")
                if attempt < self.config.MAX_RETRIES - 1:
                    time.sleep(self.config.RETRY_DELAY)
                else:
                    logger.error("Max retries reached for API error")
                    
            except Exception as e:
                logger.error(f"Unexpected error generating embedding: {str(e)}")
                break
        
        return None
    
    def generate_embeddings_batch(self, segments: List[DocumentTextSegment]) -> Dict[str, Any]:
        """Generate embeddings for multiple segments with batch optimization"""
        results = {
            'processed_segments': 0,
            'failed_segments': 0,
            'skipped_segments': 0,
            'errors': [],
            'details': []
        }
        
        if not segments:
            logger.info("No segments provided for embedding")
            return results
        
        # Filter segments that should be embedded
        embeddable_segments = []
        for segment in segments:
            should_embed, reason = self.should_embed_segment(segment)
            if should_embed:
                embeddable_segments.append(segment)
            else:
                results['skipped_segments'] += 1
                results['details'].append({
                    'segment_id': str(segment.id),
                    'status': 'skipped',
                    'reason': reason
                })
                logger.debug(f"Skipped segment {segment.id}: {reason}")
        
        if not embeddable_segments:
            logger.info("No segments to embed after filtering")
            return results
        
        logger.info(f"Generating embeddings for {len(embeddable_segments)} segments")
        
        # Process segments in batches
        for i in range(0, len(embeddable_segments), self.config.BATCH_SIZE):
            batch = embeddable_segments[i:i + self.config.BATCH_SIZE]
            batch_results = self._process_batch(batch)
            
            # Aggregate results
            results['processed_segments'] += batch_results['processed']
            results['failed_segments'] += batch_results['failed']
            results['errors'].extend(batch_results['errors'])
            results['details'].extend(batch_results['details'])
            
            # Small delay between batches to be respectful to API
            if i + self.config.BATCH_SIZE < len(embeddable_segments):
                time.sleep(0.5)
        
        logger.info(f"Batch processing complete: {results['processed_segments']} processed, {results['failed_segments']} failed, {results['skipped_segments']} skipped")
        return results
    
    def _process_batch(self, segments: List[DocumentTextSegment]) -> Dict[str, Any]:
        """Process a single batch of segments"""
        batch_results = {
            'processed': 0,
            'failed': 0,
            'errors': [],
            'details': []
        }
        
        for segment in segments:
            try:
                start_time = time.time()
                
                embedding = self.generate_embedding_with_retry(segment.content)
                
                if embedding:
                    # Save embedding to database with transaction
                    with transaction.atomic():
                        segment.embedding = embedding
                        segment.embedding_model = self.model
                        segment.save(update_fields=['embedding', 'embedding_model'])
                    
                    processing_time = time.time() - start_time
                    batch_results['processed'] += 1
                    batch_results['details'].append({
                        'segment_id': str(segment.id),
                        'status': 'success',
                        'processing_time': round(processing_time, 2)
                    })
                    
                    logger.debug(f"Generated embedding for segment {segment.id} in {processing_time:.2f}s")
                    
                else:
                    batch_results['failed'] += 1
                    error_msg = f"Failed to generate embedding for segment {segment.id}"
                    batch_results['errors'].append(error_msg)
                    batch_results['details'].append({
                        'segment_id': str(segment.id),
                        'status': 'failed',
                        'reason': 'Embedding generation failed'
                    })
                    logger.warning(error_msg)
                    
            except Exception as e:
                batch_results['failed'] += 1
                error_msg = f"Error processing segment {segment.id}: {str(e)}"
                batch_results['errors'].append(error_msg)
                batch_results['details'].append({
                    'segment_id': str(segment.id),
                    'status': 'error',
                    'reason': str(e)
                })
                logger.error(error_msg)
        
        return batch_results
    
    def generate_embeddings_for_document(self, document_id: str) -> Dict[str, Any]:
        """Generate embeddings for all segments in a document"""
        try:
            document = Document.objects.get(id=document_id)
            segments = list(document.text_segments.all().order_by('sequence_number'))
            
            if not segments:
                return {
                    'processed_segments': 0,
                    'failed_segments': 0,
                    'skipped_segments': 0,
                    'errors': ['No text segments found for document']
                }
            
            logger.info(f"Processing {len(segments)} segments for document: {document.filename}")
            
            # Use batch processing
            results = self.generate_embeddings_batch(segments)
            
            # Update document metadata if all processing is complete
            try:
                self._update_document_embedding_status(document, results)
            except Exception as e:
                logger.warning(f"Failed to update document status: {str(e)}")
            
            return results
            
        except Document.DoesNotExist:
            error_msg = f"Document {document_id} not found"
            logger.error(error_msg)
            return {
                'processed_segments': 0,
                'failed_segments': 0,
                'skipped_segments': 0,
                'errors': [error_msg]
            }
        except Exception as e:
            error_msg = f"Error processing document {document_id}: {str(e)}"
            logger.error(error_msg)
            return {
                'processed_segments': 0,
                'failed_segments': 0,
                'skipped_segments': 0,
                'errors': [error_msg]
            }
    
    def _update_document_embedding_status(self, document: Document, results: Dict[str, Any]):
        """Update document metadata with embedding status"""
        import json
        
        try:
            # Parse existing metadata
            metadata = json.loads(document.metadata) if document.metadata != '{}' else {}
            
            # Update embedding status
            total_segments = document.text_segments.count()
            segments_with_embeddings = document.text_segments.filter(embedding__isnull=False).count()
            
            metadata['embedding_status'] = {
                'total_segments': total_segments,
                'embedded_segments': segments_with_embeddings,
                'coverage': round((segments_with_embeddings / max(total_segments, 1)) * 100, 2),
                'last_processed': datetime.now().isoformat(),
                'processing_results': {
                    'processed': results['processed_segments'],
                    'failed': results['failed_segments'], 
                    'skipped': results['skipped_segments']
                }
            }
            
            # Set status based on coverage
            if segments_with_embeddings == total_segments and total_segments > 0:
                metadata['embedding_status']['status'] = 'completed'
            elif segments_with_embeddings > 0:
                metadata['embedding_status']['status'] = 'partial'
            else:
                metadata['embedding_status']['status'] = 'failed'
            
            document.metadata = json.dumps(metadata)
            document.save(update_fields=['metadata'])
            
            logger.info(f"Updated embedding status for document {document.filename}: {metadata['embedding_status']['status']}")
            
        except Exception as e:
            logger.error(f"Failed to update document metadata: {str(e)}")
    
    # Legacy method - redirects to new retry method
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Legacy method - redirects to new retry method"""
        return self.generate_embedding_with_retry(text)
    
    def semantic_search(self, query: str, limit: int = 10, document_type: Optional[str] = None) -> List[Dict]:
        """
        Perform semantic search - simplified for testing without pgvector
        """
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            
            if not query_embedding:
                return []
            
            # Fallback to simple filtering when pgvector not available
            if not PGVECTOR_AVAILABLE:
                logger.warning("pgvector not available, using simple text search fallback")
                return self._keyword_search(query, limit)
            
            # Use Django ORM with pgvector
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
            logger.error(f"Error in semantic search: {e}")
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
            logger.error(f"Error in keyword search: {e}")
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
    """
    processor = VectorProcessor()
    
    if search_type == 'semantic':
        return processor.semantic_search(query, limit)
    else:  # keyword fallback
        return processor._keyword_search(query, limit)
