"""
Test script to verify async/sync Django ORM fix
"""
import os
import sys
import asyncio

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'reportminer.settings')

import django
django.setup()

from django.db import connection
from apps.ingestion.models import Document, DocumentTextSegment

def test_database_in_thread():
    """Test database access in a separate thread"""
    try:
        # Setup Django in this thread
        django.setup()
        
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM document_text_segments")
            segment_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM document_text_segments WHERE embedding IS NOT NULL")
            embedded_count = cursor.fetchone()[0]
        
        return f"✅ SUCCESS: {doc_count} docs, {segment_count} segments, {embedded_count} embedded"
        
    except Exception as e:
        return f"❌ FAILED: {str(e)}"
    finally:
        connection.close()

async def test_async_to_sync():
    """Test running Django ORM in thread pool from async context"""
    try:
        print("🧪 Testing async/sync Django integration...")
        
        # Run database operation in thread pool
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, test_database_in_thread)
        
        print(f"📊 Database Test: {result}")
        
        # Test RAG engine
        try:
            from apps.query.rag_engine import get_rag_engine
            rag_result = await loop.run_in_executor(None, test_rag_in_thread)
            print(f"🧠 RAG Test: {rag_result}")
        except Exception as e:
            print(f"🧠 RAG Test: ❌ {str(e)}")
        
        print("✅ Async/Sync integration working!")
        return True
        
    except Exception as e:
        print(f"❌ Async/Sync test failed: {str(e)}")
        return False

def test_rag_in_thread():
    """Test RAG engine in thread"""
    try:
        django.setup()
        from apps.query.rag_engine import get_rag_engine
        
        rag = get_rag_engine()
        health = rag.health_check()
        
        return f"✅ RAG OK: {health.get('overall_status', 'unknown')}"
        
    except Exception as e:
        return f"❌ RAG Failed: {str(e)}"
    finally:
        connection.close()

def test_sync_only():
    """Test sync-only version"""
    print("🧪 Testing sync-only Django access...")
    
    try:
        # This should work in main thread
        doc_count = Document.objects.count()
        segment_count = DocumentTextSegment.objects.count()
        
        print(f"✅ Sync access: {doc_count} documents, {segment_count} segments")
        return True
        
    except Exception as e:
        print(f"❌ Sync access failed: {str(e)}")
        return False

async def main():
    """Main test function"""
    print("🚀 Testing ReportMiner Async/Sync Integration\n")
    
    # Test 1: Sync access (should work)
    print("=== Test 1: Sync Access ===")
    sync_ok = test_sync_only()
    
    print("\n=== Test 2: Async to Sync via Thread Pool ===")
    async_ok = await test_async_to_sync()
    
    print(f"\n📋 RESULTS:")
    print(f"   Sync access: {'✅ PASS' if sync_ok else '❌ FAIL'}")
    print(f"   Async/Sync: {'✅ PASS' if async_ok else '❌ FAIL'}")
    
    if sync_ok and async_ok:
        print(f"\n🎉 SUCCESS: MCP tools should work with thread pool approach!")
        print(f"💡 Use: await loop.run_in_executor(None, sync_function, args)")
    else:
        print(f"\n⚠️ ISSUES: Some tests failed, check Django/database configuration")

if __name__ == "__main__":
    asyncio.run(main())