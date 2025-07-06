"""
Tests for search functionality and integration
"""
import unittest
from django.test import TestCase
from django.test.utils import override_settings
from apps.ingestion.models import Document, DocumentTextSegment
from apps.ingestion.vector_processor import VectorProcessor
from unittest.mock import patch, MagicMock


class SemanticSearchTestCase(TestCase):
    """Test semantic search functionality"""
    
    def setUp(self):
        """Set up test documents with embeddings"""
        # Create test documents
        self.doc1 = Document.objects.create(
            filename="financial_report.pdf",
            file_type="pdf",
            document_type="financial_report"
        )
        
        self.doc2 = Document.objects.create(
            filename="technical_manual.pdf", 
            file_type="pdf",
            document_type="technical_document"
        )
        
        # Create segments with mock embeddings
        self.segments = [
            DocumentTextSegment.objects.create(
                document=self.doc1,
                sequence_number=1,
                content="The company's revenue increased by 15% this quarter due to strong sales performance.",
                segment_type="paragraph",
                embedding=[0.1] * 1536  # Mock embedding
            ),
            DocumentTextSegment.objects.create(
                document=self.doc1,
                sequence_number=2,
                content="Operating expenses were reduced by implementing cost-cutting measures.",
                segment_type="paragraph",
                embedding=[0.2] * 1536  # Mock embedding
            ),
            DocumentTextSegment.objects.create(
                document=self.doc2,
                sequence_number=1,
                content="The technical specifications require proper installation procedures.",
                segment_type="paragraph",
                embedding=[0.3] * 1536  # Mock embedding
            ),
            DocumentTextSegment.objects.create(
                document=self.doc2,
                sequence_number=2,
                content="System maintenance should be performed quarterly to ensure optimal performance.",
                segment_type="paragraph",
                embedding=[0.4] * 1536  # Mock embedding
            )
        ]
        
        self.processor = VectorProcessor()
    
    @patch('openai.OpenAI')
    def test_semantic_search_basic(self, mock_openai):
        """Test basic semantic search functionality"""
        # Mock query embedding generation
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.15] * 1536  # Similar to first segment
        
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Perform search
        results = self.processor.semantic_search("revenue financial performance", limit=5)
        
        # Should return results
        self.assertGreater(len(results), 0)
        
        # Results should have required fields
        for result in results:
            self.assertIn('segment_id', result)
            self.assertIn('content', result)
            self.assertIn('document_id', result)
            self.assertIn('filename', result)
            self.assertIn('similarity_score', result)
            
        # Should be ordered by similarity
        if len(results) > 1:
            for i in range(len(results) - 1):
                self.assertGreaterEqual(results[i]['similarity_score'], results[i+1]['similarity_score'])
    
    @patch('openai.OpenAI')
    def test_semantic_search_with_document_type_filter(self, mock_openai):
        """Test semantic search with document type filtering"""
        # Mock query embedding
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.15] * 1536
        
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Search only financial documents
        results = self.processor.semantic_search(
            "revenue performance", 
            limit=5, 
            document_type="financial_report"
        )
        
        # All results should be from financial documents
        for result in results:
            self.assertEqual(result['document_type'], "financial_report")
    
    @patch('openai.OpenAI')
    def test_search_with_no_embeddings(self, mock_openai):
        """Test search when no embeddings exist"""
        # Create document without embeddings
        doc_no_embed = Document.objects.create(
            filename="no_embed.pdf",
            file_type="pdf"
        )
        
        DocumentTextSegment.objects.create(
            document=doc_no_embed,
            sequence_number=1,
            content="Content without embeddings",
            segment_type="paragraph"
            # No embedding field set
        )
        
        # Mock query embedding
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.1] * 1536
        
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Search should only return segments with embeddings
        results = self.processor.semantic_search("content", limit=10)
        
        # Should not include the segment without embeddings
        result_content = [r['content'] for r in results]
        self.assertNotIn("Content without embeddings", result_content)
    
    def test_search_error_handling(self):
        """Test search behavior when embedding generation fails"""
        with patch('openai.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_client.embeddings.create.side_effect = Exception("API Error")
            mock_openai.return_value = mock_client
            
            # Should return empty list on error
            results = self.processor.semantic_search("test query")
            self.assertEqual(len(results), 0)


class HybridSearchTestCase(TestCase):
    """Test hybrid search functionality"""
    
    def setUp(self):
        """Set up test data for hybrid search"""
        self.document = Document.objects.create(
            filename="hybrid_test.pdf",
            file_type="pdf"
        )
        
        # Create segments with both text content and embeddings
        self.segments = [
            DocumentTextSegment.objects.create(
                document=self.document,
                sequence_number=1,
                content="Financial revenue report shows quarterly growth in sales and profit margins.",
                segment_type="paragraph",
                embedding=[0.1] * 1536
            ),
            DocumentTextSegment.objects.create(
                document=self.document,
                sequence_number=2,
                content="Technical documentation for system maintenance and operational procedures.",
                segment_type="paragraph",
                embedding=[0.2] * 1536
            ),
            DocumentTextSegment.objects.create(
                document=self.document,
                sequence_number=3,
                content="Revenue streams and financial performance metrics for Q3 analysis.",
                segment_type="paragraph",
                embedding=[0.3] * 1536
            )
        ]
        
        self.processor = VectorProcessor()
    
    @patch('openai.OpenAI')
    def test_hybrid_search_functionality(self, mock_openai):
        """Test hybrid search combining keyword and semantic search"""
        # Mock query embedding
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.15] * 1536
        
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Perform hybrid search
        results = self.processor.hybrid_search("revenue financial", limit=5)
        
        # Should return results
        self.assertGreater(len(results), 0)
        
        # Results should have hybrid search fields
        for result in results:
            self.assertIn('combined_score', result)
            self.assertIn('vector_distance', result)
            self.assertIn('keyword_rank', result)
            self.assertIn('similarity_score', result)
        
        # Should be ordered by combined score
        if len(results) > 1:
            for i in range(len(results) - 1):
                self.assertGreaterEqual(results[i]['combined_score'], results[i+1]['combined_score'])
    
    @patch('openai.OpenAI')
    def test_hybrid_search_weight_configuration(self, mock_openai):
        """Test hybrid search with different weight configurations"""
        # Mock query embedding
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.1] * 1536
        
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Test keyword-heavy search
        keyword_heavy = self.processor.hybrid_search(
            "revenue", 
            keyword_weight=0.8, 
            semantic_weight=0.2
        )
        
        # Test semantic-heavy search
        semantic_heavy = self.processor.hybrid_search(
            "revenue", 
            keyword_weight=0.2, 
            semantic_weight=0.8
        )
        
        # Both should return results but potentially in different orders
        self.assertGreater(len(keyword_heavy), 0)
        self.assertGreater(len(semantic_heavy), 0)
    
    def test_hybrid_search_fallback_to_keyword(self):
        """Test hybrid search fallback when embeddings fail"""
        with patch('openai.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_client.embeddings.create.side_effect = Exception("Embedding failed")
            mock_openai.return_value = mock_client
            
            # Should fallback to keyword search
            results = self.processor.hybrid_search("revenue")
            
            # May return results from keyword search fallback
            # Check that it doesn't crash
            self.assertIsInstance(results, list)


class KeywordSearchTestCase(TestCase):
    """Test keyword search functionality"""
    
    def setUp(self):
        """Set up test data for keyword search"""
        self.document = Document.objects.create(
            filename="keyword_test.pdf",
            file_type="pdf"
        )
        
        # Create segments for keyword search testing
        self.segments = [
            DocumentTextSegment.objects.create(
                document=self.document,
                sequence_number=1,
                content="The quarterly revenue report shows significant growth in sales.",
                segment_type="paragraph"
            ),
            DocumentTextSegment.objects.create(
                document=self.document,
                sequence_number=2,
                content="Financial performance metrics indicate strong profitability.",
                segment_type="paragraph"
            ),
            DocumentTextSegment.objects.create(
                document=self.document,
                sequence_number=3,
                content="Technical specifications for the new product line.",
                segment_type="paragraph"
            )
        ]
        
        self.processor = VectorProcessor()
    
    def test_keyword_search_basic(self):
        """Test basic keyword search functionality"""
        results = self.processor._keyword_search("revenue", limit=5)
        
        # Should find segments containing "revenue"
        self.assertGreater(len(results), 0)
        
        # All results should contain the search term
        for result in results:
            self.assertIn('revenue', result['content'].lower())
        
        # Results should have keyword rank
        for result in results:
            self.assertIn('keyword_rank', result)
            self.assertIn('search_type', result)
            self.assertEqual(result['search_type'], 'keyword_only')
    
    def test_keyword_search_multiple_terms(self):
        """Test keyword search with multiple terms"""
        results = self.processor._keyword_search("financial performance", limit=5)
        
        # Should return results
        self.assertIsInstance(results, list)
        
        # Results should be ranked by relevance
        if len(results) > 1:
            for i in range(len(results) - 1):
                self.assertGreaterEqual(results[i]['keyword_rank'], results[i+1]['keyword_rank'])
    
    def test_keyword_search_no_results(self):
        """Test keyword search with no matching results"""
        results = self.processor._keyword_search("nonexistent term", limit=5)
        
        # Should return empty list
        self.assertEqual(len(results), 0)
    
    def test_keyword_search_error_handling(self):
        """Test keyword search error handling"""
        # Test with invalid query that might cause SQL errors
        results = self.processor._keyword_search("", limit=5)
        
        # Should handle gracefully
        self.assertIsInstance(results, list)


class SearchIntegrationTestCase(TestCase):
    """Test search integration with the convenience functions"""
    
    def setUp(self):
        """Set up test data for integration testing"""
        self.document = Document.objects.create(
            filename="integration_test.pdf",
            file_type="pdf"
        )
        
        # Create segments with embeddings
        DocumentTextSegment.objects.create(
            document=self.document,
            sequence_number=1,
            content="Revenue growth and financial performance analysis for Q3 reporting.",
            segment_type="paragraph",
            embedding=[0.1] * 1536
        )
        
        DocumentTextSegment.objects.create(
            document=self.document,
            sequence_number=2,
            content="Technical documentation and system maintenance procedures.",
            segment_type="paragraph",
            embedding=[0.2] * 1536
        )
    
    def test_search_documents_convenience_function(self):
        """Test the search_documents convenience function"""
        from apps.ingestion.vector_processor import search_documents
        
        # Test keyword search
        keyword_results = search_documents("revenue", search_type="keyword", limit=5)
        self.assertIsInstance(keyword_results, list)
        
        # Test semantic search (will be mocked)
        with patch('openai.OpenAI') as mock_openai:
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = [0.1] * 1536
            
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            semantic_results = search_documents("revenue", search_type="semantic", limit=5)
            self.assertIsInstance(semantic_results, list)
        
        # Test hybrid search
        with patch('openai.OpenAI') as mock_openai:
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = [0.1] * 1536
            
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            hybrid_results = search_documents("revenue", search_type="hybrid", limit=5)
            self.assertIsInstance(hybrid_results, list)
    
    def test_generate_document_embeddings_convenience_function(self):
        """Test the generate_document_embeddings convenience function"""
        from apps.ingestion.vector_processor import generate_document_embeddings
        
        with patch('openai.OpenAI') as mock_openai:
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = [0.1] * 1536
            
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            # Test with valid document
            results = generate_document_embeddings(str(self.document.id))
            
            self.assertIn('processed_segments', results)
            self.assertIn('failed_segments', results)
            self.assertIn('skipped_segments', results)
            
            # Test with invalid document
            invalid_results = generate_document_embeddings("00000000-0000-0000-0000-000000000000")
            self.assertIn('not found', invalid_results['errors'][0])


if __name__ == '__main__':
    unittest.main()
