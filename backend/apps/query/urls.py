"""
URL configuration for query app
Maps endpoints to views
"""

from django.urls import path
from .views import (
    NaturalLanguageQueryView,
    DocumentSimilarityView,
    QueryHealthView,
    quick_query,
    batch_query,
    query_analytics
)

app_name = 'query'

urlpatterns = [
    # Main natural language query endpoint
    path('nl-query/', NaturalLanguageQueryView.as_view(), name='natural-language-query'),
    
    # Document similarity search
    path('similar/', DocumentSimilarityView.as_view(), name='document-similarity'),
    
    # Health check
    path('health/', QueryHealthView.as_view(), name='query-health'),
    
    # Quick test endpoint
    path('quick/', quick_query, name='quick-query'),
    
    # NEW: Batch processing
    path('batch/', batch_query, name='batch-query'),
    
    # NEW: Analytics endpoint
    path('analytics/', query_analytics, name='query-analytics'),
]

"""
Available endpoints after this setup:

POST /api/query/nl-query/
- Main AI query endpoint
- Body: {"query": "What financial data do you have?"}

POST /api/query/similar/ 
- Find similar documents
- Body: {"query": "revenue", "limit": 5}

GET /api/query/health/
- Check system health
- No body required

POST /api/query/quick/
- Simple test endpoint  
- Body: {"q": "test question"}

POST /api/query/batch/
- Process multiple queries at once
- Body: {"queries": ["What is revenue?", "Who are stakeholders?"], "include_sources": false}

GET /api/query/analytics/
- Get system analytics and stats
- No body required
"""