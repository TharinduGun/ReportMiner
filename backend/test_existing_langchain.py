#!/usr/bin/env python
"""
Test LangChain integration with existing ReportMiner data
FIXED VERSION - Handles embedding array truthiness issue
"""
import os
import sys
import django

# Set up Django
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'reportminer.settings')
django.setup()

from apps.ingestion.models import Document, DocumentTextSegment

def check_existing_data():
    """Check what data we currently have"""
    print("🔍 CHECKING EXISTING REPORTMINER DATA")
    print("=" * 50)
    
    # Check documents
    docs = Document.objects.all()
    print(f"📄 Total Documents: {docs.count()}")
    
    # Check text segments
    segments = DocumentTextSegment.objects.all()
    print(f"📝 Total Text Segments: {segments.count()}")
    
    # Check embeddings
    segments_with_embeddings = DocumentTextSegment.objects.filter(embedding__isnull=False)
    print(f"🔮 Segments with Embeddings: {segments_with_embeddings.count()}")
    
    return docs, segments, segments_with_embeddings

def test_with_existing_data():
    """Test LangChain components with existing data"""
    print("\n🧪 TESTING WITH EXISTING DATA")
    print("=" * 50)
    
    docs, segments, segments_with_embeddings = check_existing_data()
    
    if not docs.exists():
        print("❌ No documents found. Please upload some test documents first.")
        return False
    
    # Test document access
    print("\n📋 Document Details:")
    for doc in docs[:3]:  # Show first 3 documents
        print(f"  📄 {doc.filename} ({doc.processing_status})")
        
        # Get segments for this document
        doc_segments = doc.text_segments.all()
        segments_with_emb = doc.text_segments.filter(embedding__isnull=False)
        
        print(f"     📝 Segments: {doc_segments.count()}")
        print(f"     🔮 With Embeddings: {segments_with_emb.count()}")
        
        # Show sample segment with embedding details - FIXED VERSION
        if segments_with_emb.exists():
            segment = segments_with_emb.first()
            print(f"     📋 Sample Content: {segment.content[:100]}...")
            
            # FIX 1: Check if embedding exists and has content properly
            if segment.embedding is not None and hasattr(segment.embedding, '__len__'):
                embedding_dims = len(segment.embedding)
                print(f"     🔮 Embedding Dimensions: {embedding_dims}")
            else:
                print(f"     🔮 Embedding: None or Invalid")
    
    return True

def test_langchain_basics():
    """Test basic LangChain functionality"""
    print("\n🦜 TESTING LANGCHAIN BASICS")
    print("=" * 50)
    
    try:
        # Test basic imports
        from langchain.schema import Document as LangChainDocument
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        print("✅ LangChain imports successful")
        
        # Test with sample data
        segments_with_embeddings = DocumentTextSegment.objects.filter(embedding__isnull=False)
        
        if segments_with_embeddings.exists():
            sample_segment = segments_with_embeddings.first()
            
            # Create LangChain document
            langchain_doc = LangChainDocument(
                page_content=sample_segment.content,
                metadata={
                    "source": sample_segment.document.filename,
                    "segment_id": str(sample_segment.id),
                    "sequence": sample_segment.sequence_number
                }
            )
            
            print(f"✅ Created LangChain Document")
            print(f"   Content Length: {len(langchain_doc.page_content)}")
            print(f"   Metadata: {langchain_doc.metadata}")
            
        else:
            print("⚠️ No segments with embeddings found for LangChain testing")
            
    except ImportError as e:
        print(f"❌ LangChain import failed: {e}")
        print("💡 Install with: pip install langchain")
        return False
    except Exception as e:
        print(f"❌ LangChain test failed: {e}")
        return False
    
    return True

def test_vector_operations():
    """Test vector operations with existing embeddings"""
    print("\n🔮 TESTING VECTOR OPERATIONS")
    print("=" * 50)
    
    segments_with_embeddings = DocumentTextSegment.objects.filter(embedding__isnull=False)
    
    if not segments_with_embeddings.exists():
        print("❌ No embeddings found. Cannot test vector operations.")
        return False
    
    print(f"✅ Found {segments_with_embeddings.count()} segments with embeddings")
    
    # Test embedding format
    sample_segment = segments_with_embeddings.first()
    
    # FIX 2: Safer embedding checking
    embedding = sample_segment.embedding
    if embedding is not None:
        try:
            if hasattr(embedding, '__len__'):
                print(f"✅ Embedding format: {type(embedding)} with {len(embedding)} dimensions")
                
                # Check if it's a proper vector (list of numbers)
                if isinstance(embedding, list) and len(embedding) > 0:
                    if isinstance(embedding[0], (int, float)):
                        print(f"✅ Valid embedding vector: [{embedding[0]:.4f}, {embedding[1]:.4f}, ...]")
                    else:
                        print(f"⚠️ Embedding contains non-numeric data: {type(embedding[0])}")
                else:
                    print(f"⚠️ Embedding is not a proper list: {embedding}")
            else:
                print(f"⚠️ Embedding has no length attribute: {type(embedding)}")
                
        except Exception as e:
            print(f"❌ Error inspecting embedding: {e}")
    else:
        print("❌ Embedding is None")
    
    return True

def test_postgresql_vector():
    """Test PostgreSQL vector operations"""
    print("\n🐘 TESTING POSTGRESQL VECTOR OPERATIONS")
    print("=" * 50)
    
    try:
        from django.db import connection
        
        with connection.cursor() as cursor:
            # Test pgvector extension
            cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
            result = cursor.fetchone()
            
            if result:
                print("✅ pgvector extension is installed")
                
                # Test vector operations
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM document_text_segments 
                    WHERE embedding IS NOT NULL
                """)
                count = cursor.fetchone()[0]
                print(f"✅ Found {count} segments with vector embeddings in database")
                
                if count > 0:
                    # Test similarity search
                    cursor.execute("""
                        SELECT id, content, 
                               embedding <-> (SELECT embedding FROM document_text_segments WHERE embedding IS NOT NULL LIMIT 1) as distance
                        FROM document_text_segments 
                        WHERE embedding IS NOT NULL 
                        ORDER BY distance 
                        LIMIT 3
                    """)
                    
                    results = cursor.fetchall()
                    print(f"✅ Vector similarity search successful:")
                    for i, (seg_id, content, distance) in enumerate(results, 1):
                        print(f"   {i}. Distance: {distance:.4f} | Content: {content[:60]}...")
                
            else:
                print("❌ pgvector extension not found")
                return False
                
    except Exception as e:
        print(f"❌ PostgreSQL vector test failed: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    print("🚀 REPORTMINER LANGCHAIN INTEGRATION TEST")
    print("=" * 60)
    
    success_count = 0
    total_tests = 4
    
    # Run tests
    if test_with_existing_data():
        success_count += 1
    
    if test_langchain_basics():
        success_count += 1
    
    if test_vector_operations():
        success_count += 1
    
    if test_postgresql_vector():
        success_count += 1
    
    # Results
    print(f"\n🏁 TEST RESULTS")
    print("=" * 30)
    print(f"✅ Passed: {success_count}/{total_tests}")
    print(f"❌ Failed: {total_tests - success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("\n🎉 ALL TESTS PASSED! Ready for LangChain integration.")
        print("\n📋 NEXT STEPS:")
        print("1. Install LangChain packages: pip install langchain langchain-postgres langchain-openai")
        print("2. Create LangChain vector store wrapper")
        print("3. Implement RAG query system")
        print("4. Test natural language queries")
    else:
        print(f"\n⚠️ {total_tests - success_count} tests failed. Fix issues before proceeding.")
    
    return success_count == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)