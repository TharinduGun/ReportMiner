#!/usr/bin/env python
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'reportminer.settings')
django.setup()

from django.conf import settings

print("🔗 Testing Connection Strings")
print("=" * 30)

db_config = settings.DATABASES['default']

# LangChain psycopg3 format  
langchain = f"postgresql+psycopg://{db_config['USER']}:{db_config['PASSWORD']}@{db_config['HOST']}:{db_config['PORT']}/{db_config['NAME']}"
print("LangChain format:", langchain)

print("📝 This is the connection string to use for LangChain")