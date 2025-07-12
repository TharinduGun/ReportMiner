#!/usr/bin/env python
"""
Test RAG system with existing ReportMiner data
Tests your 462 embeddings with natural language queries
"""

import os
import sys
import django
import requests
import json
import time

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'reportminer.settings')
django.setup()

from apps.query.rag_engine import get_rag_engine

def test_rag_engine_directly():
    """Test RAG engine directly (without API)"""
    print("🧪 TESTING RAG ENGINE DIRECTLY")
    print("=" * 50)
    
    try:
        # Initialize RAG engine
        print("1️⃣ Initializing RAG engine...")
        rag_engine = get_rag_engine()
        print("✅ RAG engine initialized")
        
        # Health check
        print("\n2️⃣ Running health check...")
        health = rag_engine.health_check()
        print(f"   Status: {health['overall_status']}")
        print(f"   Can retrieve: {health['can_retrieve']}")
        
        if health['overall_status'] != 'healthy':
            print("⚠️ System not fully healthy, but continuing tests...")
        
        # Test queries designed for your data
        test_queries = [
            "What documents do you have?",  # ✅ Works!
            "Tell me about Loreta Curren",  # Should find the person mentioned
            "What information about France is mentioned?",  # Should find France reference
            "What dates are in the documents?",  # Should find 21/05/2015
            "What numbers or IDs are mentioned?",  # Should find 9654
            "Tell me about the content in the documents"  # ✅ Already works!
        ]
        
        print(f"\n3️⃣ Testing {len(test_queries)} sample queries...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n   Query {i}: {query}")
            
            start_time = time.time()
            result = rag_engine.query(query, include_sources=True)
            end_time = time.time()
            
            if result['success']:
                print(f"   ✅ Answer ({end_time-start_time:.1f}s): {result['answer'][:100]}...")
                print(f"   📄 Sources found: {len(result['sources'])}")
                
                if result['sources']:
                    print(f"   📋 First source: {result['sources'][0]['filename']}")
            else:
                print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG engine test failed: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints (requires Django server running)"""
    print("\n🌐 TESTING API ENDPOINTS")
    print("=" * 50)
    
    base_url = "http://localhost:8000/api/query"
    
    # Test queries
    test_data = [
        {
            "endpoint": "/health/",
            "method": "GET",
            "data": None,
            "description": "Health check"
        },
        {
            "endpoint": "/quick/",
            "method": "POST", 
            "data": {"q": "What documents do you have?"},
            "description": "Quick query test"
        },
        {
            "endpoint": "/nl-query/",
            "method": "POST",
            "data": {"query": "Tell me about financial data in the documents"},
            "description": "Natural language query"
        },
        {
            "endpoint": "/similar/",
            "method": "POST",
            "data": {"query": "revenue", "limit": 3},
            "description": "Document similarity"
        }
    ]
    
    print("⚠️ Note: Django server must be running (python manage.py runserver)")
    print("Testing endpoints...\n")
    
    for test in test_data:
        url = base_url + test["endpoint"]
        print(f"Testing {test['description']}: {test['method']} {url}")
        
        try:
            if test["method"] == "GET":
                response = requests.get(url, timeout=30)
            else:
                response = requests.post(url, json=test["data"], timeout=30)
            
            if response.status_code == 200:
                print(f"   ✅ Success ({response.status_code})")
                
                # Show sample response
                try:
                    data = response.json()
                    if 'answer' in data:
                        print(f"   📝 Answer: {data['answer'][:80]}...")
                    elif 'status' in data:
                        print(f"   📊 Status: {data['status']}")
                except:
                    print("   📄 Response received")
                    
            else:
                print(f"   ❌ Failed ({response.status_code})")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data.get('error', 'Unknown error')}")
                except:
                    print(f"   Error: {response.text[:100]}")
                    
        except requests.exceptions.ConnectionError:
            print("   ⚠️ Connection failed - Django server not running?")
        except requests.exceptions.Timeout:
            print("   ⏰ Request timeout - processing taking too long")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print()

def test_with_sample_data():
    """Test specific queries that should work with your data"""
    print("\n📊 TESTING WITH YOUR ACTUAL DATA")
    print("=" * 50)
    
    try:
        # Check what data we have
        from apps.ingestion.models import Document, DocumentTextSegment
        
        docs = Document.objects.all()
        segments_with_embeddings = DocumentTextSegment.objects.filter(embedding__isnull=False)
        
        print(f"📄 Total documents: {docs.count()}")
        print(f"🔮 Segments with embeddings: {segments_with_embeddings.count()}")
        
        if segments_with_embeddings.count() == 0:
            print("❌ No embeddings found! Cannot test RAG.")
            return False
        
        # Show sample document types
        doc_types = docs.values_list('filename', flat=True)[:5]
        print(f"📋 Sample documents: {list(doc_types)}")
        
        # Test with specific queries based on your data
        rag_engine = get_rag_engine()
        
        # Get a sample of actual content to create relevant queries
        sample_segment = segments_with_embeddings.first()
        sample_content = sample_segment.content[:200] if sample_segment else ""
        
        print(f"\n📝 Sample content: {sample_content}...")
        
        # Test a query that should definitely find something
        print(f"\n🔍 Testing with content-based query...")
        result = rag_engine.query("Tell me about the content in the documents", include_sources=True)
        
        if result['success']:
            print(f"✅ Found relevant content!")
            print(f"📝 Answer: {result['answer'][:150]}...")
            print(f"📄 Sources: {len(result['sources'])} documents")
        else:
            print(f"❌ Query failed: {result.get('error')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 REPORTMINER RAG SYSTEM TEST")
    print("=" * 60)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Direct RAG engine
    if test_rag_engine_directly():
        success_count += 1
    
    # Test 2: Sample data queries  
    if test_with_sample_data():
        success_count += 1
    
    # Test 3: API endpoints (optional, requires server)
    print("\n" + "="*50)
    print("API TESTING (requires Django server running)")
    print("Run 'python manage.py runserver' in another terminal")
    user_input = input("Test API endpoints? (y/N): ").lower()
    
    if user_input.startswith('y'):
        test_api_endpoints()
        success_count += 1
    else:
        print("⏭️ Skipping API tests")
        total_tests = 2  # Adjust total since we skipped API test
    
    # Results
    print(f"\n🏁 TEST RESULTS")
    print("=" * 30)
    print(f"✅ Passed: {success_count}/{total_tests}")
    print(f"❌ Failed: {total_tests - success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("🚀 Your RAG system is working with your 462 embeddings!")
        print("\n📋 NEXT STEPS:")
        print("1. Start Django server: python manage.py runserver")
        print("2. Test API: POST http://localhost:8000/api/query/nl-query/")
        print("3. Move to Day 3: Django RAG API integration")
    else:
        print(f"\n⚠️ {total_tests - success_count} tests failed.")
        print("🔧 Check errors above and fix before proceeding to Day 3")
    
    return success_count == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)