"""
RAG (Retrieval-Augmented Generation) Engine for ReportMiner
Uses your existing 462 embeddings to answer questions intelligently
Manual implementation to bypass LangChain retriever validation issues
"""

import logging
from typing import Dict, List, Any, Optional
from django.conf import settings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document as LangChainDocument
from apps.ingestion.langchain_wrapper import ReportMinerLangChainWrapper
from apps.ingestion.vector_processor import VectorProcessor
from apps.ingestion.models import Document, DocumentTextSegment


logger = logging.getLogger(__name__)

class RAGQueryEngine:
    """
    Core RAG engine that answers questions using your existing embeddings
    
    How it works:
    1. Takes a natural language question
    2. Finds relevant documents from your 462 embeddings using custom vector search
    3. Uses GPT to generate intelligent answers based on found documents
    4. Returns answer with source attribution
    
    Uses manual RAG implementation for better control and compatibility
    """
    
    def __init__(self):
        """Initialize RAG components"""
        logger.info("🧠 Initializing RAG Query Engine...")
        
        # Connect to your existing embeddings (no data migration needed!)
        self.wrapper = ReportMinerLangChainWrapper()
        
        # Verify we have embeddings to work with
        stats = self.wrapper.get_stats()
        logger.info(f"📊 Using {stats.get('embedded_segments', 0)} existing embeddings")
        
        # Initialize OpenAI for generating answers
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0.1,
            max_tokens=500
        )
        
        # Create custom prompt
        self.prompt = self._create_custom_prompt()
        
        # Initialize manual RAG chain
        self.qa_chain = None
        self._setup_qa_chain()
        
        logger.info("✅ RAG Query Engine initialized with existing embeddings")
    
    def _create_custom_prompt(self) -> PromptTemplate:
        """
        Create a custom prompt template for better answers
        
        Why custom prompt: 
        - Ensures answers are based on documents
        - Provides source attribution
        - Handles cases where no relevant info is found
        """
        template = """You are an AI assistant helping users query their document database.

Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context, just say that you don't have enough information in the documents to answer that question.

Context from documents:
{context}

Question: {question}

Instructions:
1. Base your answer strictly on the provided context
2. If the context contains relevant information, provide a comprehensive answer
3. Include specific details, numbers, or quotes when available
4. If no relevant information is found, clearly state this
5. Keep your answer concise but informative

Answer:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _setup_qa_chain(self):
        """Setup manual QA processing (bypassing LangChain retriever issues)"""
        try:
            # Test that we can retrieve documents using our custom implementation
            test_docs = self.wrapper.test_similarity_search("test", k=1)
            if test_docs:
                self.qa_chain = "manual_rag"  # Use manual RAG implementation
                logger.info("✅ Manual RAG Chain setup completed")
            else:
                self.qa_chain = None
                logger.error("❌ No documents retrievable")
                
        except Exception as e:
            logger.error(f"❌ Error setting up QA chain: {e}")
            self.qa_chain = None
    
    def query(self, question: str, include_sources: bool = True) -> Dict[str, Any]:
        """
        Main query method using manual RAG implementation
        
        Args:
            question: Natural language question
            include_sources: Whether to return source documents
            
        Returns:
            Dict with answer, sources, and metadata
        """
        if not question or not question.strip():
            return {
                "success": False,
                "error": "Question cannot be empty",
                "answer": None,
                "sources": []
            }
        
        if self.qa_chain != "manual_rag":
            return {
                "success": False,
                "error": "RAG system not properly initialized",
                "answer": None,
                "sources": []
            }
        
        try:
            logger.info(f"🔍 Processing question: {question}")
            
            # Step 1: Retrieve relevant documents using your existing embeddings
            relevant_docs = self.wrapper.test_similarity_search(question, k=5)
            
            if not relevant_docs:
                return {
                    "success": True,
                    "answer": "I don't have enough information in the documents to answer that question.",
                    "sources": [],
                    "metadata": {
                        "question": question,
                        "sources_found": 0,
                        "answer_quality": "no_sources",
                        "method": "manual_rag"
                    }
                }
            
            # Step 2: Create context from retrieved documents
            context_parts = []
            for i, doc in enumerate(relevant_docs):
                context_parts.append(f"Document {i+1}: {doc['content']}")
            
            context = "\n\n".join(context_parts)
            
            # Step 3: Create prompt for LLM using our template
            prompt_text = self.prompt.format(context=context, question=question)
            
            # Step 4: Generate answer using OpenAI
            answer = self.llm.invoke(prompt_text).content
            
            # Step 5: Format sources for response
            formatted_sources = []
            if include_sources:
                formatted_sources = self._format_sources_from_search_results(relevant_docs)
            
            # Step 6: Assess answer quality
            answer_quality = self._assess_answer_quality(answer, relevant_docs)
            
            response = {
                "success": True,
                "answer": answer,
                "sources": formatted_sources,
                "metadata": {
                    "question": question,
                    "sources_found": len(relevant_docs),
                    "answer_quality": answer_quality,
                    "model_used": "gpt-4o-mini",
                    "method": "manual_rag"
                }
            }
            
            logger.info(f"✅ Question answered successfully with {len(relevant_docs)} sources")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error processing question: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": None,
                "sources": []
            }
    
    def _format_sources_from_search_results(self, search_results: List[Dict]) -> List[Dict[str, Any]]:
        """Format search results for API response"""
        formatted = []
        
        for i, result in enumerate(search_results):
            source = {
                "source_number": i + 1,
                "content": result["content"],
                "filename": result["metadata"].get("filename", "Unknown"),
                "segment_id": result["metadata"].get("segment_id"),
                "sequence_number": result["metadata"].get("sequence_number"),
                "segment_type": result["metadata"].get("segment_type", "paragraph"),
                "distance": result.get("distance", "N/A")
            }
            
            # Add section info if available
            if result["metadata"].get("section_title"):
                source["section_title"] = result["metadata"]["section_title"]
            
            formatted.append(source)
        
        return formatted
    
    def _format_sources(self, source_docs: List[LangChainDocument]) -> List[Dict[str, Any]]:
        """Format source documents for API response (legacy compatibility)"""
        formatted = []
        
        for i, doc in enumerate(source_docs):
            source = {
                "source_number": i + 1,
                "content": doc.page_content,
                "filename": doc.metadata.get("filename", "Unknown"),
                "segment_id": doc.metadata.get("segment_id"),
                "sequence_number": doc.metadata.get("sequence_number"),
                "segment_type": doc.metadata.get("segment_type", "paragraph"),
                "word_count": doc.metadata.get("word_count", 0)
            }
            
            # Add section info if available
            if doc.metadata.get("section_title"):
                source["section_title"] = doc.metadata["section_title"]
            
            formatted.append(source)
        
        return formatted
    
    def _assess_answer_quality(self, answer: str, source_docs: List) -> str:
        """Assess the quality of the generated answer"""
        if not answer or len(answer.strip()) < 10:
            return "poor"
        
        if "don't have enough information" in answer.lower() or "don't know" in answer.lower():
            return "insufficient_data"
        
        if len(source_docs) == 0:
            return "no_sources"
        
        if len(source_docs) >= 3 and len(answer) > 50:
            return "excellent"
        elif len(source_docs) >= 1 and len(answer) > 30:
            return "good"
        else:
            return "fair"
    
    def get_similar_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Get similar documents without generating an answer
        Useful for exploring what documents are relevant
        """
        try:
            results = self.wrapper.test_similarity_search(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Error getting similar documents: {e}")
            return []
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check if RAG engine is working properly
        """
        checks = {
            "wrapper_initialized": self.wrapper is not None,
            "llm_initialized": self.llm is not None,
            "qa_chain_initialized": self.qa_chain == "manual_rag",
            "can_retrieve": False,
            "overall_status": "unhealthy"
        }
        
        try:
            # Test retrieval
            test_docs = self.wrapper.test_similarity_search("test", k=1)
            checks["can_retrieve"] = len(test_docs) > 0
            
            # Overall status
            if all([checks["wrapper_initialized"], checks["llm_initialized"], 
                   checks["qa_chain_initialized"], checks["can_retrieve"]]):
                checks["overall_status"] = "healthy"
                
        except Exception as e:
            checks["error"] = str(e)
        
        return checks


# Convenience function for easy import
def get_rag_engine() -> RAGQueryEngine:
    """Get initialized RAG engine instance"""
    return RAGQueryEngine()