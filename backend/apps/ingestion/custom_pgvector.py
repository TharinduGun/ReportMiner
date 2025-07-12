from typing import Any, Dict, List, Optional, Tuple
from langchain_core.documents import Document as LangChainDocument
from langchain_core.embeddings import Embeddings
from langchain_postgres.vectorstores import PGVector
from django.db import connection
from .models import DocumentTextSegment, Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List

class ReportMinerPGVector:
    """Custom PGVector that reads from your existing document_text_segments table"""
    
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
    
    def similarity_search(self, query: str, k: int = 4) -> List[LangChainDocument]:
        """Search using your existing embeddings"""
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Convert to PostgreSQL vector format
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        # Query your existing table
        sql = """
        SELECT 
            ts.id,
            ts.content,
            ts.sequence_number,
            ts.section_title,
            ts.segment_type,
            d.filename,
            d.id as document_id,
            ts.embedding <-> %s as distance
        FROM document_text_segments ts
        JOIN documents d ON ts.document_id = d.id
        WHERE ts.embedding IS NOT NULL
        ORDER BY ts.embedding <-> %s
        LIMIT %s
        """
        
        with connection.cursor() as cursor:
            cursor.execute(sql, [embedding_str, embedding_str, k])
            results = cursor.fetchall()
        
        # Convert to LangChain Documents
        docs = []
        for row in results:
            doc = LangChainDocument(
                page_content=row[1],  # content
                metadata={
                    "segment_id": str(row[0]),
                    "sequence_number": row[2],
                    "section_title": row[3] or "",
                    "segment_type": row[4],
                    "filename": row[5],
                    "document_id": str(row[6]),
                    "distance": float(row[7])
                }
            )
            docs.append(doc)
        
        return docs
    
    def as_retriever(self, search_type: str = "similarity", search_kwargs: Dict = None):
        """Return a retriever interface that properly inherits from BaseRetriever"""
        if search_kwargs is None:
            search_kwargs = {"k": 4}
        
        class CustomRetriever(BaseRetriever):
            """Properly inherits from LangChain's BaseRetriever"""
            
            def __init__(self, vector_store, k):
                super().__init__()
                self.vector_store = vector_store
                self.k = k
            
            def _get_relevant_documents(
                self, 
                query: str, 
                *, 
                run_manager: CallbackManagerForRetrieverRun
            ) -> List[LangChainDocument]:
                """Required method for BaseRetriever"""
                return self.vector_store.similarity_search(query, k=self.k)
            
            # Legacy method for compatibility
            def get_relevant_documents(self, query: str) -> List[LangChainDocument]:
                """Legacy compatibility method"""
                return self.vector_store.similarity_search(query, k=self.k)
        
        return CustomRetriever(self, search_kwargs.get("k", 4))