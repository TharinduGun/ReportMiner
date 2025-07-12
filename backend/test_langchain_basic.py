#!/usr/bin/env python
"""
Test LangChain with ReportMiner data
"""
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'reportminer.settings')
django.setup()

from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from django.conf import settings

def test_langchain_with_data():
    print("🧪 Testing LangChain with ReportMiner Data")
    print("=" * 45)
    
    try:
        # Step 1: Initialize embeddings
        print("1️⃣ Initializing OpenAI embeddings...")
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=settings.OPENAI_API_KEY
        )
        print("✅ Embeddings initialized")
        
        # Step 2: Build connection string
        print("2️⃣ Building connection string...")
        db_config = settings.DATABASES['default']
        connection_string = f"postgresql+psycopg://{db_config['USER']}:{db_config['PASSWORD']}@{db_config['HOST']}:{db_config['PORT']}/{db_config['NAME']}"
        print("✅ Connection string ready")
        
        # Step 3: Try to initialize PGVector (this is the test)
        print("3️⃣ Initializing PGVector...")
        vector_store = PGVector(
            embeddings=embeddings,
            collection_name="test_collection",
            connection=connection_string,
            use_jsonb=True
        )
        print("✅ PGVector initialized successfully!")
        
        # Step 4: Test with a simple document
        print("4️⃣ Testing with sample document...")
        test_docs = [
            Document(
                page_content="This is a test document about financial revenue and profit margins.",
                metadata={"source": "test", "type": "financial"}
            )
        ]
        
        # Add documents (this tests the full pipeline)
        ids = vector_store.add_documents(test_docs)
        print(f"✅ Added {len(ids)} test documents")
        
        # Step 5: Test similarity search
        print("5️⃣ Testing similarity search...")
        results = vector_store.similarity_search("revenue financial", k=2)
        print(f"✅ Found {len(results)} similar documents")
        
        for i, doc in enumerate(results, 1):
            print(f"   {i}. {doc.page_content[:100]}...")
        
        print("\n🎉 LangChain integration test SUCCESSFUL!")
        print("✅ Ready for Day 1 completion!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("📝 Don't worry, we can fix this in the next step")
        return False

if __name__ == "__main__":
    test_langchain_with_data()