#!/usr/bin/env python
"""
Test RAG Engine Directly - Run this first to verify your RAG works
Save as: backend/test_rag_direct.py
"""
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'reportminer.settings')
django.setup()

from apps.query.rag_engine import get_rag_engine

def test_rag_engine():
    """Test RAG engine functionality"""
    print("🧪 Testing RAG Engine Directly")
    print("=" * 50)
    
    try:
        # Initialize RAG engine
        print("1️⃣ Initializing RAG engine...")
        rag = get_rag_engine()
        print("✅ RAG engine initialized")
        
        # Health check
        print("\n2️⃣ Running health check...")
        health = rag.health_check()
        print(f"Health Status: {health}")
        
        if health['overall_status'] != 'healthy':
            print("❌ RAG engine not healthy")
            return False
        
        # Test query
        print("\n3️⃣ Testing sample query...")
        test_question = "What documents do you have?"
        result = rag.query(test_question, include_sources=True)
        
        print(f"Query: {test_question}")
        print(f"Success: {result.get('success', False)}")
        print(f"Answer: {result.get('answer', 'No answer')}")
        print(f"Sources: {len(result.get('sources', []))}")
        
        if result.get('success'):
            print("✅ RAG query successful!")
            return True
        else:
            print(f"❌ RAG query failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_rag_engine()
    if success:
        print("\n🎉 RAG engine working! Ready to test MCP integration.")
    else:
        print("\n⚠️ Fix RAG engine before testing MCP.")
    sys.exit(0 if success else 1)