"""
Embedding Configuration for ReportMiner
Centralized configuration for embedding generation
"""
from django.conf import settings
import os

class EmbeddingConfig:
    # Feature flags
    ENABLE_AUTO_EMBEDDINGS = getattr(settings, 'ENABLE_AUTO_EMBEDDINGS', True)
    ENABLE_BATCH_PROCESSING = getattr(settings, 'ENABLE_BATCH_PROCESSING', True)
    ENABLE_ASYNC_EMBEDDINGS = getattr(settings, 'ENABLE_ASYNC_EMBEDDINGS', False)
    
    # Processing limits
    MAX_SEGMENTS_SYNC = getattr(settings, 'MAX_SEGMENTS_SYNC', 10)
    MIN_TEXT_LENGTH = getattr(settings, 'EMBEDDING_MIN_TEXT_LENGTH', 50)
    MAX_TEXT_LENGTH = getattr(settings, 'EMBEDDING_MAX_TEXT_LENGTH', 8000)
    BATCH_SIZE = getattr(settings, 'EMBEDDING_BATCH_SIZE', 5)
    
    # Performance settings
    REQUEST_TIMEOUT = getattr(settings, 'EMBEDDING_REQUEST_TIMEOUT', 30)
    MAX_RETRIES = getattr(settings, 'EMBEDDING_MAX_RETRIES', 3)
    RETRY_DELAY = getattr(settings, 'EMBEDDING_RETRY_DELAY', 1)
    
    # Cost control
    MAX_DAILY_API_CALLS = getattr(settings, 'MAX_DAILY_EMBEDDING_CALLS', 10000)
    COST_ALERT_THRESHOLD = getattr(settings, 'EMBEDDING_COST_ALERT', 100)  # segments
    
    # Segment filtering
    SKIP_SEGMENT_TYPES = getattr(settings, 'SKIP_EMBEDDING_TYPES', ['header', 'footer'])
    PRIORITY_SEGMENT_TYPES = getattr(settings, 'PRIORITY_EMBEDDING_TYPES', ['paragraph', 'heading'])

    