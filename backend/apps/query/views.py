from django.shortcuts import render

# Create your views here.
"""
API Views for Natural Language Querying
Exposes RAG functionality via REST endpoints
"""

import logging
import time
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from django.http import JsonResponse
from .rag_engine import RAGQueryEngine, get_rag_engine

logger = logging.getLogger(__name__)

class NaturalLanguageQueryView(APIView):
    """
    Main API endpoint for natural language queries
    
    POST /api/query/nl-query/
    Body: {"query": "What financial data do you have?"}
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rag_engine = None
    
    def _get_rag_engine(self):
        """Lazy initialization of RAG engine"""
        if self.rag_engine is None:
            try:
                self.rag_engine = get_rag_engine()
                logger.info("✅ RAG engine initialized for API")
            except Exception as e:
                logger.error(f"❌ Failed to initialize RAG engine: {e}")
                raise
        return self.rag_engine
    
    def post(self, request):
        """
        Process natural language query
        
        Expected input:
        {
            "query": "What is the total revenue mentioned in the documents?",
            "include_sources": true,  // optional, default true
            "max_sources": 5         // optional, default 5
        }
        """
        try:
            # Extract query from request
            query = request.data.get('query', '').strip()
            include_sources = request.data.get('include_sources', True)
            max_sources = min(request.data.get('max_sources', 5), 10)  # Cap at 10
            
            # Validate input
            if not query:
                return Response({
                    "success": False,
                    "error": "Query parameter is required",
                    "example": {
                        "query": "What financial information is available?"
                    }
                }, status=status.HTTP_400_BAD_REQUEST)
            
            if len(query) > 500:
                return Response({
                    "success": False,
                    "error": "Query too long (max 500 characters)",
                    "query_length": len(query)
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Get RAG engine
            rag_engine = self._get_rag_engine()
            
            # Process the query
            logger.info(f"🔍 Processing query: {query[:100]}...")
            result = rag_engine.query(query, include_sources=include_sources)
            
            if result['success']:
                # Limit sources if requested
                if include_sources and len(result['sources']) > max_sources:
                    result['sources'] = result['sources'][:max_sources]
                    result['metadata']['sources_limited'] = True
                
                return Response({
                    "success": True,
                    "query": query,
                    "answer": result['answer'],
                    "sources": result['sources'] if include_sources else [],
                    "metadata": result['metadata']
                }, status=status.HTTP_200_OK)
            
            else:
                return Response({
                    "success": False,
                    "query": query,
                    "error": result.get('error', 'Unknown error occurred'),
                    "suggestion": "Try rephrasing your question or check if documents are properly processed"
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
        except Exception as e:
            logger.error(f"❌ Error in natural language query: {e}")
            return Response({
                "success": False,
                "error": "Internal server error",
                "details": str(e) if logger.isEnabledFor(logging.DEBUG) else None
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class DocumentSimilarityView(APIView):
    """
    Find documents similar to a query without generating answers
    
    POST /api/query/similar/
    Body: {"query": "revenue", "limit": 5}
    """
    
    def post(self, request):
        try:
            query = request.data.get('query', '').strip()
            limit = min(request.data.get('limit', 5), 20)  # Cap at 20
            
            if not query:
                return Response({
                    "error": "Query parameter is required"
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Get similar documents
            rag_engine = get_rag_engine()
            similar_docs = rag_engine.get_similar_documents(query, k=limit)
            
            return Response({
                "success": True,
                "query": query,
                "similar_documents": similar_docs,
                "total_found": len(similar_docs)
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"❌ Error in similarity search: {e}")
            return Response({
                "success": False,
                "error": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class QueryHealthView(APIView):
    """
    Health check for query system
    
    GET /api/query/health/
    """
    
    def get(self, request):
        try:
            rag_engine = get_rag_engine()
            health_status = rag_engine.health_check()
            
            http_status = status.HTTP_200_OK if health_status['overall_status'] == 'healthy' else status.HTTP_503_SERVICE_UNAVAILABLE
            
            return Response({
                "service": "ReportMiner Query API",
                "status": health_status['overall_status'],
                "checks": health_status,
                "timestamp": f"{time.time():.0f}"
            }, status=http_status)
            
        except Exception as e:
            return Response({
                "service": "ReportMiner Query API",
                "status": "error",
                "error": str(e)
            }, status=status.HTTP_503_SERVICE_UNAVAILABLE)


# Function-based view for simple testing
@api_view(['POST'])
def quick_query(request):
    """
    Simple endpoint for quick testing
    
    POST /api/query/quick/
    Body: {"q": "What documents do you have?"}
    """
    try:
        query = request.data.get('q', '').strip()
        
        if not query:
            return JsonResponse({
                "error": "Parameter 'q' is required",
                "example": "POST with {'q': 'What financial data do you have?'}"
            }, status=400)
        
        # Quick query processing
        rag_engine = get_rag_engine()
        result = rag_engine.query(query, include_sources=False)
        
        if result['success']:
            return JsonResponse({
                "question": query,
                "answer": result['answer'],
                "sources_used": result['metadata']['sources_found']
            })
        else:
            return JsonResponse({
                "question": query,
                "error": result.get('error', 'Unknown error')
            }, status=500)
            
    except Exception as e:
        return JsonResponse({
            "error": str(e)
        }, status=500)


# Add new endpoint for batch queries
@api_view(['POST'])
def batch_query(request):
    """
    Process multiple queries at once
    
    POST /api/query/batch/
    Body: {
        "queries": ["What is revenue?", "Who are the stakeholders?"],
        "include_sources": false
    }
    """
    try:
        queries = request.data.get('queries', [])
        include_sources = request.data.get('include_sources', False)
        
        if not isinstance(queries, list) or len(queries) == 0:
            return Response({
                "success": False,
                "error": "queries must be a non-empty list",
                "example": {
                    "queries": ["What is revenue?", "Who are stakeholders?"],
                    "include_sources": False
                }
            }, status=status.HTTP_400_BAD_REQUEST)
        
        if len(queries) > 10:
            return Response({
                "success": False,
                "error": "Maximum 10 queries allowed per batch"
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Process each query
        rag_engine = get_rag_engine()
        results = []
        
        for i, query in enumerate(queries):
            try:
                result = rag_engine.query(query, include_sources=include_sources)
                results.append({
                    "query_index": i,
                    "query": query,
                    "success": result['success'],
                    "answer": result.get('answer'),
                    "sources_count": len(result.get('sources', [])),
                    "metadata": result.get('metadata', {})
                })
            except Exception as e:
                results.append({
                    "query_index": i,
                    "query": query,
                    "success": False,
                    "error": str(e)
                })
        
        return Response({
            "success": True,
            "total_queries": len(queries),
            "results": results
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({
            "success": False,
            "error": str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# Add endpoint for query analytics
@api_view(['GET'])
def query_analytics(request):
    """
    Get analytics about query system usage
    
    GET /api/query/analytics/
    """
    try:
        from apps.ingestion.models import Document, DocumentTextSegment
        
        # Basic stats
        stats = {
            "database_stats": {
                "total_documents": Document.objects.count(),
                "embedded_segments": DocumentTextSegment.objects.filter(
                    embedding__isnull=False
                ).count(),
                "total_segments": DocumentTextSegment.objects.count()
            },
            "system_stats": {
                "rag_engine_available": True,
                "embedding_dimensions": 1536,
                "supported_queries": [
                    "Natural language questions about document content",
                    "Search for specific information",
                    "Document comparison and analysis",
                    "Data extraction and summarization"
                ]
            }
        }
        
        # Test RAG engine availability
        try:
            rag_engine = get_rag_engine()
            health = rag_engine.health_check()
            stats["rag_health"] = health
        except Exception as e:
            stats["rag_health"] = {"status": "error", "error": str(e)}
        
        return Response({
            "success": True,
            "analytics": stats
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({
            "success": False,
            "error": str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)