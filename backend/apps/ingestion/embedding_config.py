"""
Embedding Configuration for ReportMiner - COST OPTIMIZED
"""
from django.conf import settings
import os

class EmbeddingConfig:
    # Feature flags
    ENABLE_AUTO_EMBEDDINGS = getattr(settings, 'ENABLE_AUTO_EMBEDDINGS', True)
    ENABLE_BATCH_PROCESSING = getattr(settings, 'ENABLE_BATCH_PROCESSING', True)
    
    # Processing limits - COST OPTIMIZED
    MAX_SEGMENTS_SYNC = getattr(settings, 'MAX_SEGMENTS_SYNC', 5)
    MIN_TEXT_LENGTH = getattr(settings, 'EMBEDDING_MIN_TEXT_LENGTH', 100)
    MAX_TEXT_LENGTH = getattr(settings, 'EMBEDDING_MAX_TEXT_LENGTH', 8000)
    BATCH_SIZE = getattr(settings, 'EMBEDDING_BATCH_SIZE', 3)
    
    # Performance settings
    REQUEST_TIMEOUT = getattr(settings, 'EMBEDDING_REQUEST_TIMEOUT', 30)
    MAX_RETRIES = getattr(settings, 'EMBEDDING_MAX_RETRIES', 2)
    RETRY_DELAY = getattr(settings, 'EMBEDDING_RETRY_DELAY', 1)
    
    # Cost control - STRICT LIMITS
    MAX_DAILY_API_CALLS = getattr(settings, 'MAX_DAILY_EMBEDDING_CALLS', 300)
    COST_ALERT_THRESHOLD = getattr(settings, 'EMBEDDING_COST_ALERT', 100)
    
    # Segment filtering - MORE RESTRICTIVE
    SKIP_SEGMENT_TYPES = getattr(settings, 'SKIP_EMBEDDING_TYPES', ['header', 'footer', 'list_item'])
    PRIORITY_SEGMENT_TYPES = getattr(settings, 'PRIORITY_EMBEDDING_TYPES', ['paragraph', 'heading'])
    
    # Document limits
    MAX_SEGMENTS_PER_DOCUMENT = getattr(settings, 'MAX_SEGMENTS_PER_DOCUMENT', 50)
    SKIP_LARGE_DOCUMENTS = getattr(settings, 'SKIP_LARGE_DOCUMENTS', True)