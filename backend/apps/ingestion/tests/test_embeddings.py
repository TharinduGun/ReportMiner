"""
Comprehensive tests for embedding functionality
"""
import unittest
from unittest.mock import patch, MagicMock, call
from django.test import TestCase, override_settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.cache import cache
from apps.ingestion.models import Document, DocumentTextSegment, FileUpload
from apps.ingestion.vector_processor import VectorProcessor, EmbeddingError
from apps.ingestion.enhanced_upload_pipeline import EnhancedUploadPipeline
import tempfile
import os
import openai


class EmbeddingFilteringTestCase(TestCase):
    """Test segment filtering logic"""
    
    def setUp(self):
        """Set up test data"""
        self.vector_processor = VectorProcessor()
        
        # Create test document
        self.document = Document.objects.create(
            filename="test_doc.pdf",
            original_filename="test_doc.pdf",
            file_type="pdf",
            processing_status="processing"
        )
        
        # Create test segments with different characteristics
        self.segments = [
            DocumentTextSegment.objects.create(
                document=self.document,
                sequence_number=1,
                content="This is a test paragraph with enough content to generate embeddings successfully.",
                segment_type="paragraph"
            ),
            DocumentTextSegment.objects.create(
                document=self.document,
                sequence_number=2,
                content="Short",  # Too short for embedding
                segment_type="paragraph"
            ),
            DocumentTextSegment.objects.create(
                document=self.document,
                sequence_number=3,
                content="Header content that should be skipped",
                segment_type="header"  # Should be skipped
            ),
            DocumentTextSegment.objects.create(
                document=self.document,
                sequence_number=4,
                content="",  # Empty content
                segment_type="paragraph"
            ),
            DocumentTextSegment.objects.create(
                document=self.document,
                sequence_number=5,
                content="This segment already has an embedding and should be skipped.",
                segment_type="paragraph",
                embedding=[0.1] * 1536  # Already has embedding
            )
        ]
    
    def test_should_embed_normal_paragraph(self):
        """Test that normal paragraphs should be embedded"""
        should_embed, reason = self.vector_processor.should_embed_segment(self.segments[0])
        self.assertTrue(should_embed)
        self.assertEqual(reason, "OK")
    
    def test_should_skip_short_content(self):
        """Test that short content is skipped"""
        should_embed, reason = self.vector_processor.should_embed_segment(self.segments[1])
        self.assertFalse(should_embed)
        self.assertIn("too short", reason.lower())
    
    def test_should_skip_header_type(self):
        """Test that header segments are skipped"""
        should_embed, reason = self.vector_processor.should_embed_segment(self.segments[2])
        self.assertFalse(should_embed)
        self.assertIn("header", reason.lower())
    
    def test_should_skip_empty_content(self):
        """Test that empty content is skipped"""
        should_embed, reason = self.vector_processor.should_embed_segment(self.segments[3])
        self.assertFalse(should_embed)
        self.assertIn("too short", reason.lower())
    
    def test_should_skip_existing_embedding(self):
        """Test that segments with existing embeddings are skipped"""
        should_embed, reason = self.vector_processor.should_embed_segment(self.segments[4])
        self.assertFalse(should_embed)
        self.assertIn("already has embedding", reason.lower())
    
    def test_rate_limit_checking(self):
        """Test rate limit checking functionality"""
        # Test normal operation
        self.assertTrue(self.vector_processor.check_rate_limits())
        
        # Test when limit is reached
        cache.set(self.vector_processor.daily_call_count_key, 10000)
        self.assertFalse(self.vector_processor.check_rate_limits())
        
        # Clean up
        cache.delete(self.vector_processor.daily_call_count_key)


class EmbeddingGenerationTestCase(TestCase):
    """Test embedding generation with mocked OpenAI"""
    
    def setUp(self):
        self.vector_processor = VectorProcessor()
    
    def test_text_cleaning(self):
        """Test text cleaning functionality"""
        dirty_text = "  This    has   excessive   whitespace  \n\n\n  "
        cleaned = self.vector_processor.clean_text(dirty_text)
        self.assertEqual(cleaned, "This has excessive whitespace")
        
        # Test empty text
        self.assertEqual(self.vector_processor.clean_text(""), "")
        self.assertEqual(self.vector_processor.clean_text(None), "")
        
        # Test truncation
        long_text = "A" * 10000
        cleaned = self.vector_processor.clean_text(long_text)
        self.assertLessEqual(len(cleaned), 8000)
    
    @patch('openai.OpenAI')
    def test_successful_embedding_generation(self, mock_openai):
        """Test successful embedding generation"""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.1] * 1536  # Valid embedding
        
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Test embedding generation
        embedding = self.vector_processor.generate_embedding_with_retry("Test content for embedding")
        
        self.assertIsNotNone(embedding)
        self.assertEqual(len(embedding), 1536)
        self.assertEqual(embedding, [0.1] * 1536)
        
        # Verify API was called
        mock_client.embeddings.create.assert_called_once()
    
    @patch('openai.OpenAI')
    def test_invalid_embedding_dimensions(self, mock_openai):
        """Test handling of invalid embedding dimensions"""
        # Mock OpenAI response with wrong dimensions
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.1] * 512  # Wrong dimensions
        
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Should return None for invalid dimensions
        embedding = self.vector_processor.generate_embedding_with_retry("Test content")
        self.assertIsNone(embedding)
    
    @patch('openai.OpenAI')
    def test_openai_rate_limit_handling(self, mock_openai):
        """Test rate limit error handling with retries"""
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = openai.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body=None
        )
        mock_openai.return_value = mock_client
        
        # Should return None after retries
        with patch('time.sleep') as mock_sleep:  # Speed up test
            embedding = self.vector_processor.generate_embedding_with_retry("Test content")
            
        self.assertIsNone(embedding)
        # Should have tried 3 times (MAX_RETRIES)
        self.assertEqual(mock_client.embeddings.create.call_count, 3)
        # Should have slept between retries
        self.assertEqual(mock_sleep.call_count, 2)
    
    @patch('openai.OpenAI')
    def test_openai_timeout_handling(self, mock_openai):
        """Test timeout error handling"""
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = openai.APITimeoutError(
            message="Request timed out",
            request=MagicMock()
        )
        mock_openai.return_value = mock_client
        
        with patch('time.sleep'):  # Speed up test
            embedding = self.vector_processor.generate_embedding_with_retry("Test content")
            
        self.assertIsNone(embedding)
        self.assertEqual(mock_client.embeddings.create.call_count, 3)
    
    @patch('openai.OpenAI')
    def test_openai_api_error_handling(self, mock_openai):
        """Test general API error handling"""
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = openai.APIError(
            message="API Error",
            request=MagicMock(),
            body=None
        )
        mock_openai.return_value = mock_client
        
        with patch('time.sleep'):  # Speed up test
            embedding = self.vector_processor.generate_embedding_with_retry("Test content")
            
        self.assertIsNone(embedding)
        self.assertEqual(mock_client.embeddings.create.call_count, 3)
    
    @patch('openai.OpenAI')
    def test_retry_success_after_failure(self, mock_openai):
        """Test successful retry after initial failure"""
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.1] * 1536
        
        mock_client = MagicMock()
        # First call fails, second succeeds
        mock_client.embeddings.create.side_effect = [
            openai.APITimeoutError(message="Timeout", request=MagicMock()),
            mock_response
        ]
        mock_openai.return_value = mock_client
        
        with patch('time.sleep'):  # Speed up test
            embedding = self.vector_processor.generate_embedding_with_retry("Test content")
            
        self.assertIsNotNone(embedding)
        self.assertEqual(len(embedding), 1536)
        self.assertEqual(mock_client.embeddings.create.call_count, 2)


class BatchProcessingTestCase(TestCase):
    """Test batch processing functionality"""
    
    def setUp(self):
        """Create test document with multiple segments"""
        self.document = Document.objects.create(
            filename="batch_test_doc.pdf",
            original_filename="batch_test_doc.pdf",
            file_type="pdf"
        )
        
        # Create 10 test segments
        self.segments = []
        for i in range(10):
            segment = DocumentTextSegment.objects.create(
                document=self.document,
                sequence_number=i,
                content=f"This is test paragraph number {i} with sufficient content for embedding generation.",
                segment_type="paragraph"
            )
            self.segments.append(segment)
        
        self.vector_processor = VectorProcessor()
    
    @patch('openai.OpenAI')
    def test_successful_batch_processing(self, mock_openai):
        """Test successful batch processing of multiple segments"""
        # Mock OpenAI to return valid embeddings
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.1] * 1536
        
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Process batch
        results = self.vector_processor.generate_embeddings_batch(self.segments)
        
        # Verify results
        self.assertEqual(results['processed_segments'], 10)
        self.assertEqual(results['failed_segments'], 0)
        self.assertEqual(results['skipped_segments'], 0)
        self.assertEqual(len(results['errors']), 0)
        
        # Verify all segments have embeddings
        for segment in self.segments:
            segment.refresh_from_db()
            self.assertIsNotNone(segment.embedding)
    
    @patch('openai.OpenAI')
    def test_partial_batch_failure(self, mock_openai):
        """Test batch processing with some failures"""
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.1] * 1536
        
        mock_client = MagicMock()
        # Alternate between success and failure
        mock_client.embeddings.create.side_effect = [
            mock_response,  # Success
            openai.APIError(message="API Error", request=MagicMock(), body=None),  # Fail
            mock_response,  # Success
            openai.APIError(message="API Error", request=MagicMock(), body=None),  # Fail
            mock_response,  # Success
            mock_response,  # Success
            mock_response,  # Success
            mock_response,  # Success
            mock_response,  # Success
            mock_response,  # Success
        ]
        mock_openai.return_value = mock_client
        
        with patch('time.sleep'):  # Speed up test
            results = self.vector_processor.generate_embeddings_batch(self.segments)
        
        # Should have some successes and some failures
        self.assertEqual(results['processed_segments'], 8)  # 8 successes
        self.assertEqual(results['failed_segments'], 2)     # 2 failures
        self.assertEqual(results['skipped_segments'], 0)
        self.assertGreater(len(results['errors']), 0)
    
    def test_empty_batch_processing(self):
        """Test batch processing with no segments"""
        results = self.vector_processor.generate_embeddings_batch([])
        
        self.assertEqual(results['processed_segments'], 0)
        self.assertEqual(results['failed_segments'], 0)
        self.assertEqual(results['skipped_segments'], 0)
    
    @override_settings(EMBEDDING_BATCH_SIZE=3)
    def test_batch_size_configuration(self):
        """Test that batch size configuration is respected"""
        # This test verifies that processing respects batch size
        # We'll count API calls to ensure batching works
        with patch('openai.OpenAI') as mock_openai, \
             patch('time.sleep') as mock_sleep:
            
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = [0.1] * 1536
            
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            # Process batch
            self.vector_processor.generate_embeddings_batch(self.segments)
            
            # With 10 segments and batch size 3, we should process in 4 batches
            # This means 3 sleep calls between batches
            self.assertEqual(mock_sleep.call_count, 3)


class DocumentProcessingTestCase(TestCase):
    """Test full document embedding processing"""
    
    def setUp(self):
        self.document = Document.objects.create(
            filename="integration_test.pdf",
            file_type="pdf",
            metadata='{"test": true}'
        )
        
        # Create mixed segments
        self.segments = [
            DocumentTextSegment.objects.create(
                document=self.document,
                sequence_number=1,
                content="This is a good paragraph for embedding.",
                segment_type="paragraph"
            ),
            DocumentTextSegment.objects.create(
                document=self.document,
                sequence_number=2,
                content="Header Text",
                segment_type="header"  # Should be skipped
            ),
            DocumentTextSegment.objects.create(
                document=self.document,
                sequence_number=3,
                content="Another good paragraph for embedding generation.",
                segment_type="paragraph"
            )
        ]
        
        self.vector_processor = VectorProcessor()
    
    @patch('openai.OpenAI')
    def test_document_embedding_generation(self, mock_openai):
        """Test generating embeddings for an entire document"""
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.1] * 1536
        
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Process document
        results = self.vector_processor.generate_embeddings_for_document(str(self.document.id))
        
        # Should process 2 paragraphs, skip 1 header
        self.assertEqual(results['processed_segments'], 2)
        self.assertEqual(results['skipped_segments'], 1)
        self.assertEqual(results['failed_segments'], 0)
        
        # Check document metadata was updated
        self.document.refresh_from_db()
        self.assertIn('embedding_status', self.document.metadata)
    
    def test_nonexistent_document(self):
        """Test handling of nonexistent document"""
        fake_id = "00000000-0000-0000-0000-000000000000"
        results = self.vector_processor.generate_embeddings_for_document(fake_id)
        
        self.assertEqual(results['processed_segments'], 0)
        self.assertIn('not found', results['errors'][0])
    
    def test_document_with_no_segments(self):
        """Test document with no text segments"""
        empty_doc = Document.objects.create(
            filename="empty_doc.pdf",
            file_type="pdf"
        )
        
        results = self.vector_processor.generate_embeddings_for_document(str(empty_doc.id))
        
        self.assertEqual(results['processed_segments'], 0)
        self.assertIn('No text segments found', results['errors'][0])


class IntegrationTestCase(TestCase):
    """Test full pipeline integration with embeddings"""
    
    def setUp(self):
        """Set up test file upload"""
        self.test_content = """
TEST DOCUMENT

This is a test paragraph with sufficient content for embedding generation.
It contains multiple sentences and should be processed successfully.

FINANCIAL SUMMARY
Revenue: $1,000,000
Profit: $200,000
Growth Rate: 15%

This is another paragraph that should also receive embeddings.
The content is long enough to be meaningful for semantic search.
        """
        
        # Create test file upload
        test_file = SimpleUploadedFile(
            "test_integration.txt",
            self.test_content.encode(),
            content_type="text/plain"
        )
        
        self.file_upload = FileUpload.objects.create(
            filename="test_integration.txt",
            file=test_file,
            type="txt"
        )
    
    @patch('openai.OpenAI')
    def test_full_pipeline_with_embeddings(self, mock_openai):
        """Test complete upload pipeline with embedding generation"""
        # Mock OpenAI
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.1] * 1536
        
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Process through pipeline
        pipeline = EnhancedUploadPipeline()
        result = pipeline.process_uploaded_file(self.file_upload)
        
        # Verify success
        self.assertTrue(result['success'])
        self.assertIsNotNone(result['document_id'])
        
        # Check embedding results
        processing_results = result['processing_results']
        self.assertIn('embeddings_generated', processing_results)
        self.assertGreater(processing_results.get('embeddings_generated', 0), 0)
        
        # Verify document creation and segments
        document = Document.objects.get(id=result['document_id'])
        segments_with_embeddings = document.text_segments.exclude(embedding__isnull=True)
        self.assertGreater(segments_with_embeddings.count(), 0)
        
        # Verify that embeddings are proper vectors
        for segment in segments_with_embeddings:
            self.assertEqual(len(segment.embedding), 1536)
            self.assertIsInstance(segment.embedding, list)
    
    @patch('openai.OpenAI')
    def test_pipeline_with_embedding_failures(self, mock_openai):
        """Test pipeline behavior when embeddings fail"""
        # Mock OpenAI to always fail
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = openai.APIError(
            message="API Error",
            request=MagicMock(),
            body=None
        )
        mock_openai.return_value = mock_client
        
        # Process through pipeline
        pipeline = EnhancedUploadPipeline()
        
        with patch('time.sleep'):  # Speed up test
            result = pipeline.process_uploaded_file(self.file_upload)
        
        # Should still succeed overall (embeddings are optional)
        self.assertTrue(result['success'])
        
        # But embedding results should show failures
        processing_results = result['processing_results']
        self.assertEqual(processing_results.get('embeddings_generated', 0), 0)
        self.assertGreater(len(processing_results.get('errors', [])), 0)


class CacheAndRateLimitTestCase(TestCase):
    """Test caching and rate limiting functionality"""
    
    def setUp(self):
        self.vector_processor = VectorProcessor()
        # Clear cache before each test
        cache.clear()
    
    def test_call_count_increment(self):
        """Test that API call counting works correctly"""
        initial_count = cache.get(self.vector_processor.daily_call_count_key, 0)
        
        # Increment call count
        self.vector_processor.increment_call_count()
        
        new_count = cache.get(self.vector_processor.daily_call_count_key, 0)
        self.assertEqual(new_count, initial_count + 1)
    
    def test_rate_limit_enforcement(self):
        """Test that rate limits are properly enforced"""
        # Set count near limit
        cache.set(self.vector_processor.daily_call_count_key, 9999)
        self.assertTrue(self.vector_processor.check_rate_limits())
        
        # Set count at limit
        cache.set(self.vector_processor.daily_call_count_key, 10000)
        self.assertFalse(self.vector_processor.check_rate_limits())
        
        # Set count over limit
        cache.set(self.vector_processor.daily_call_count_key, 10001)
        self.assertFalse(self.vector_processor.check_rate_limits())
    
    def tearDown(self):
        # Clean up cache after each test
        cache.clear()


if __name__ == '__main__':
    unittest.main()
