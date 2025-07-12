#!/usr/bin/env python
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'reportminer.settings')
django.setup()

print("🧪 Testing LangChain Imports")
print("=" * 30)

try:
    from langchain_openai import OpenAIEmbeddings
    print("✅ langchain_openai imported")
    
    from langchain_postgres import PGVector
    print("✅ langchain_postgres imported")
    
    from langchain_core.documents import Document
    print("✅ langchain_core imported")
    
    from django.conf import settings
    if hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=settings.OPENAI_API_KEY
        )
        print("✅ OpenAI embeddings initialized")
    else:
        print("⚠️ No OpenAI API key found")
    
    doc = Document(
        page_content="This is a test document",
        metadata={"source": "test"}
    )
    print("✅ LangChain Document created")
    
    print("🎉 All LangChain components working!")
    
except Exception as e:
    print(f"❌ Error: {e}")