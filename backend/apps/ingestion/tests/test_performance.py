"""
Performance tests for embedding generation
"""
import time
import unittest
from django.test import TestCase, override_settings
from django.test.utils import override_settings
from apps.ingestion.models import Document, DocumentTextSegment
from apps.ingestion.vector_processor import VectorProcessor
from unittest.mock import patch, MagicMock


class EmbeddingPerformanceTestCase(TestCase):
    """Test performance characteristics of embedding generation"""
    
    def setUp(self):
        """Create test document with many segments"""
        self.document = Document.objects.create(
            filename="large_test_doc.pdf",
            original_filename="large_test_doc.pdf",
            file_type="pdf"
        )
        
        # Create 20 test segments for performance testing
        self.segments = []
        for i in range(20):
            segment = DocumentTextSegment.objects.create(
                document=self.document,
                sequence_number=i,
                content=f"This is test paragraph number {i} with sufficient content for embedding generation. " * 3,
                segment_type="paragraph"
            )
            self.segments.append(segment)
    
    @patch('openai.OpenAI')
    def test_batch_processing_performance(self, mock_openai):
        """Test performance of batch processing"""
        # Mock OpenAI to return quickly
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.1] * 1536
        
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        processor = VectorProcessor()
        
        start_time = time.time()
        results = processor.generate_embeddings_batch(self.segments)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should process all segments
        self.assertEqual(results['processed_segments'], 20)
        self.assertEqual(results['failed_segments'], 0)
        
        # Should complete in reasonable time (adjust based on your requirements)
        self.assertLess(processing_time, 30)  # Should complete within 30 seconds
        
        # Log performance metrics
        print(f"\n=== PERFORMANCE METRICS ===")
        print(f"Processed {results['processed_segments']} segments in {processing_time:.2f} seconds")
        print(f"Average time per segment: {processing_time/20:.2f} seconds")
        print(f"API calls made: {mock_client.embeddings.create.call_count}")
    
    @override_settings(EMBEDDING_BATCH_SIZE=3)
    @patch('openai.OpenAI')
    def test_small_batch_performance(self, mock_openai):
        """Test performance with smaller batch sizes"""
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.1] * 1536
        
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        processor = VectorProcessor()
        
        start_time = time.time()
        results = processor.generate_embeddings_batch(self.segments)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        print(f"\n=== SMALL BATCH PERFORMANCE ===")
        print(f"Processed {results['processed_segments']} segments in {processing_time:.2f} seconds")
        print(f"Batch size: 3")
        print(f"Expected batches: {len(self.segments) // 3 + (1 if len(self.segments) % 3 else 0)}")
    
    @override_settings(EMBEDDING_BATCH_SIZE=10)
    @patch('openai.OpenAI')
    def test_large_batch_performance(self, mock_openai):
        """Test performance with larger batch sizes"""
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.1] * 1536
        
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        processor = VectorProcessor()
        
        start_time = time.time()
        results = processor.generate_embeddings_batch(self.segments)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        print(f"\n=== LARGE BATCH PERFORMANCE ===")
        print(f"Processed {results['processed_segments']} segments in {processing_time:.2f} seconds")
        print(f"Batch size: 10")
    
    def test_memory_usage_large_document(self):
        """Test memory efficiency with large documents"""
        # Create a document with many segments to test memory usage
        large_doc = Document.objects.create(
            filename="memory_test_doc.pdf",
            file_type="pdf"
        )
        
        # Create 100 segments
        large_segments = []
        for i in range(100):
            segment = DocumentTextSegment.objects.create(
                document=large_doc,
                sequence_number=i,
                content=f"Memory test paragraph {i} " * 20,  # Longer content
                segment_type="paragraph"
            )
            large_segments.append(segment)
        
        # Test that we can process without memory issues
        processor = VectorProcessor()
        
        # Mock to avoid actual API calls
        with patch('openai.OpenAI') as mock_openai:
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = [0.1] * 1536
            
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            # This should complete without memory errors
            results = processor.generate_embeddings_batch(large_segments)
            
            self.assertEqual(results['processed_segments'], 100)
            print(f"\n=== MEMORY TEST ===")
            print(f"Successfully processed 100 segments without memory issues")


class StressTestCase(TestCase):
    """Stress tests for embedding system"""
    
    def setUp(self):
        """Set up stress test environment"""
        self.processor = VectorProcessor()
    
    @patch('openai.OpenAI')
    def test_concurrent_document_processing(self, mock_openai):
        """Test processing multiple documents simultaneously"""
        # Mock OpenAI
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.1] * 1536
        
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Create multiple documents
        documents = []
        for i in range(5):
            doc = Document.objects.create(
                filename=f"stress_doc_{i}.pdf",
                file_type="pdf"
            )
            
            # Add segments to each document
            for j in range(10):
                DocumentTextSegment.objects.create(
                    document=doc,
                    sequence_number=j,
                    content=f"Stress test content for doc {i}, segment {j} " * 5,
                    segment_type="paragraph"
                )
            
            documents.append(doc)
        
        # Process all documents
        start_time = time.time()
        results_list = []
        
        for doc in documents:
            results = self.processor.generate_embeddings_for_document(str(doc.id))
            results_list.append(results)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify all documents processed
        total_processed = sum(r['processed_segments'] for r in results_list)
        total_failed = sum(r['failed_segments'] for r in results_list)
        
        self.assertEqual(total_processed, 50)  # 5 docs * 10 segments each
        self.assertEqual(total_failed, 0)
        
        print(f"\n=== STRESS TEST ===")
        print(f"Processed 5 documents (50 segments) in {processing_time:.2f} seconds")
        print(f"Average time per document: {processing_time/5:.2f} seconds")
    
    @patch('openai.OpenAI')
    def test_error_recovery_under_load(self, mock_openai):
        """Test system behavior when API errors occur under load"""
        # Mock OpenAI to fail intermittently
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.1] * 1536
        
        import openai
        mock_client = MagicMock()
        
        # Create a pattern of successes and failures
        responses = []
        for i in range(30):
            if i % 4 == 0:  # Every 4th call fails
                responses.append(openai.APIError(
                    message="Intermittent API Error",
                    request=MagicMock(),
                    body=None
                ))
            else:
                responses.append(mock_response)
        
        mock_client.embeddings.create.side_effect = responses
        mock_openai.return_value = mock_client
        
        # Create document with many segments
        doc = Document.objects.create(filename="error_recovery_test.pdf", file_type="pdf")
        segments = []
        for i in range(30):
            segment = DocumentTextSegment.objects.create(
                document=doc,
                sequence_number=i,
                content=f"Error recovery test segment {i} " * 3,
                segment_type="paragraph"
            )
            segments.append(segment)
        
        # Process with errors
        with patch('time.sleep'):  # Speed up retry delays
            results = self.processor.generate_embeddings_batch(segments)
        
        # Should have processed most segments despite errors
        self.assertGreater(results['processed_segments'], 20)  # At least 20/30 should succeed
        self.assertGreater(results['failed_segments'], 0)      # Some should fail
        
        print(f"\n=== ERROR RECOVERY TEST ===")
        print(f"Processed {results['processed_segments']}/{len(segments)} segments with intermittent errors")
        print(f"Failed segments: {results['failed_segments']}")
        print(f"Error rate: {(results['failed_segments']/len(segments))*100:.1f}%")


class ConfigurationTestCase(TestCase):
    """Test different configuration scenarios"""
    
    def setUp(self):
        self.processor = VectorProcessor()
        # Create test document and segments
        self.document = Document.objects.create(
            filename="config_test.pdf",
            file_type="pdf"
        )
        
        self.segments = []
        for i in range(10):
            segment = DocumentTextSegment.objects.create(
                document=self.document,
                sequence_number=i,
                content=f"Configuration test segment {i} with adequate content for embedding.",
                segment_type="paragraph"
            )
            self.segments.append(segment)
    
    @override_settings(EMBEDDING_MIN_TEXT_LENGTH=100)
    def test_min_length_filtering(self):
        """Test minimum text length filtering"""
        # Create segment with short content
        short_segment = DocumentTextSegment.objects.create(
            document=self.document,
            sequence_number=999,
            content="Short text",  # Less than 100 chars
            segment_type="paragraph"
        )
        
        should_embed, reason = self.processor.should_embed_segment(short_segment)
        self.assertFalse(should_embed)
        self.assertIn("too short", reason.lower())
    
    @override_settings(SKIP_EMBEDDING_TYPES=['paragraph', 'heading'])
    def test_segment_type_filtering(self):
        """Test segment type filtering configuration"""
        # With paragraph in skip list, should be skipped
        should_embed, reason = self.processor.should_embed_segment(self.segments[0])
        self.assertFalse(should_embed)
        self.assertIn("paragraph", reason.lower())
    
    @override_settings(EMBEDDING_MAX_RETRIES=1)
    @patch('openai.OpenAI')
    def test_retry_configuration(self, mock_openai):
        """Test retry configuration"""
        import openai
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = openai.APIError(
            message="Test error",
            request=MagicMock(),
            body=None
        )
        mock_openai.return_value = mock_client
        
        with patch('time.sleep'):  # Speed up test
            embedding = self.processor.generate_embedding_with_retry("Test content")
        
        # Should only try once (MAX_RETRIES=1)
        self.assertIsNone(embedding)
        self.assertEqual(mock_client.embeddings.create.call_count, 1)
    
    @override_settings(MAX_DAILY_EMBEDDING_CALLS=5)
    def test_rate_limit_configuration(self):
        """Test daily rate limit configuration"""
        from django.core.cache import cache
        
        # Set count to limit
        cache.set(self.processor.daily_call_count_key, 5)
        
        # Should be blocked
        self.assertFalse(self.processor.check_rate_limits())
        
        # Reset for other tests
        cache.delete(self.processor.daily_call_count_key)


if __name__ == '__main__':
    unittest.main()
