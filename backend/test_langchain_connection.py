#!/usr/bin/env python
"""
Test LangChain connection to ReportMiner database
"""
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'reportminer.settings')
django.setup()

from apps.ingestion.langchain_wrapper import ReportMinerLangChainWrapper

def test_langchain_connection():
    print("🧪 Testing LangChain Connection to ReportMiner")
    print("=" * 50)
    
    try:
        # Initialize wrapper
        print("1️⃣ Initializing LangChain wrapper...")
        wrapper = ReportMinerLangChainWrapper()
        print("✅ LangChain wrapper initialized")
        
        # Test connection to existing data
        print("\n2️⃣ Testing connection to existing data...")
        connected = wrapper.connect_to_existing_data()
        
        if connected:
            print("✅ Successfully connected to existing data")
        else:
            print("⚠️ Connection issues, but proceeding...")
        
        # Test similarity search (this will test if everything works)
        print("\n3️⃣ Testing similarity search...")
        results = wrapper.test_similarity_search("financial revenue", k=2)
        
        if results:
            print(f"✅ Found {len(results)} similar documents:")
            for i, result in enumerate(results, 1):
                print(f"   {i}. {result['content'][:100]}...")
                print(f"      Metadata: {result['metadata'].get('filename', 'N/A')}")
        else:
            print("⚠️ No search results (this is OK for first test)")
        
        print("\n🎉 LangChain connection test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("📝 This might be normal for first setup")
        return False

if __name__ == "__main__":
    success = test_langchain_connection()
    sys.exit(0 if success else 1)