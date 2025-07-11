"""
LangChain Wrapper for ReportMiner - Uses existing embeddings directly
"""

import os
from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document as LangChainDocument
from django.conf import settings
from django.db import connection
from .models import Document, DocumentTextSegment
from .custom_pgvector import ReportMinerPGVector

class ReportMinerLangChainWrapper:
    """
    LangChain wrapper that uses your existing 462 embeddings directly
    No data migration needed!
    """
    
    def __init__(self):
        """Initialize with existing embeddings"""
        print("🔗 Connecting to existing ReportMiner embeddings...")
        
        # Initialize embeddings (same model you used)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        # Use custom vector store that reads your existing table
        self.vector_store = ReportMinerPGVector(self.embeddings)
        
        # Verify connection
        self._verify_existing_data()
    
    def _verify_existing_data(self) -> bool:
        """Verify we can access existing embeddings"""
        try:
            # Count existing embeddings
            segments_with_embeddings = DocumentTextSegment.objects.filter(
                embedding__isnull=False
            ).count()
            
            print(f"✅ Found {segments_with_embeddings} existing embeddings")
            
            if segments_with_embeddings == 0:
                print("⚠️ No embeddings found in document_text_segments table")
                return False
            
            # Test a sample search
            sample_docs = self.vector_store.similarity_search("test", k=1)
            print(f"✅ Successfully retrieved {len(sample_docs)} sample documents")
            
            return True
            
        except Exception as e:
            print(f"❌ Error verifying existing data: {e}")
            return False
    
    def test_similarity_search(self, query: str, k: int = 3) -> List[Dict]:
        """Test similarity search with existing embeddings"""
        try:
            print(f"🔍 Searching existing embeddings for: '{query}'")
            
            docs = self.vector_store.similarity_search(query, k=k)
            
            results = []
            for doc in docs:
                results.append({
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata,
                    "distance": doc.metadata.get("distance", "N/A")
                })
            
            print(f"✅ Found {len(results)} relevant documents")
            return results
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def get_retriever(self, search_type: str = "similarity", search_kwargs: Dict = None):
        """Get retriever for RAG chain"""
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about existing embeddings"""
        try:
            with connection.cursor() as cursor:
                # Count embeddings by document
                cursor.execute("""
                    SELECT 
                        d.filename,
                        COUNT(ts.id) as segment_count
                    FROM documents d
                    LEFT JOIN document_text_segments ts ON d.id = ts.document_id 
                        AND ts.embedding IS NOT NULL
                    GROUP BY d.id, d.filename
                    ORDER BY segment_count DESC
                """)
                
                doc_stats = cursor.fetchall()
                
                # Total counts
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_segments,
                        COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as embedded_segments
                    FROM document_text_segments
                """)
                
                total_stats = cursor.fetchone()
                
                return {
                    "total_documents": len(doc_stats),
                    "total_segments": total_stats[0],
                    "embedded_segments": total_stats[1],
                    "documents_with_embeddings": [(row[0], row[1]) for row in doc_stats if row[1] > 0]
                }
                
        except Exception as e:
            return {"error": str(e)}

# Convenience function
def get_langchain_wrapper() -> ReportMinerLangChainWrapper:
    """Get initialized wrapper using existing embeddings"""
    return ReportMinerLangChainWrapper()