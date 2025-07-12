#!/usr/bin/env python
"""
Step 1: Verify current ReportMiner setup for LangChain integration
"""
import os
import sys
import django

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(backend_dir, 'backend'))

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'reportminer.settings')
django.setup()

from django.db import connection
from apps.ingestion.models import Document, DocumentTextSegment

def check_pgvector():
    print("🔍 Checking pgvector installation...")
    
    try:
        with connection.cursor() as cursor:
            # Check if pgvector extension exists
            cursor.execute("SELECT * FROM pg_available_extensions WHERE name = 'vector';")
            available = cursor.fetchone()
            
            if available:
                print("✅ pgvector extension is available")
                
                # Check if it's installed
                cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
                installed = cursor.fetchone()
                
                if installed:
                    print("✅ pgvector extension is installed")
                    return True
                else:
                    print("⚠️ pgvector extension is available but not installed")
                    # Try to install
                    try:
                        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                        print("✅ pgvector extension installed successfully")
                        return True
                    except Exception as e:
                        print(f"❌ Failed to install pgvector: {e}")
                        return False
            else:
                print("❌ pgvector extension is not available")
                return False
                
    except Exception as e:
        print(f"❌ Error checking pgvector: {e}")
        return False

def check_existing_data():
    print("\n📊 Checking existing ReportMiner data...")
    
    try:
        # Check documents
        doc_count = Document.objects.count()
        print(f"✅ Documents in database: {doc_count}")
        
        # Check text segments
        segment_count = DocumentTextSegment.objects.count()
        print(f"✅ Text segments in database: {segment_count}")
        
        # Check segments with embeddings
        embedded_count = DocumentTextSegment.objects.filter(embedding__isnull=False).count()
        print(f"✅ Segments with embeddings: {embedded_count}")
        
        if embedded_count > 0:
            # Test vector field
            sample_segment = DocumentTextSegment.objects.filter(embedding__isnull=False).first()
            if sample_segment and sample_segment.embedding:
                print(f"✅ Sample embedding dimensions: {len(sample_segment.embedding)}")
                return True
            else:
                print("⚠️ Embedding field exists but no valid embeddings found")
                return False
        else:
            print("⚠️ No embeddings found in database")
            return False
            
    except Exception as e:
        print(f"❌ Error checking existing data: {e}")
        return False

def check_database_schema():
    print("\n🗄️ Checking database schema...")
    
    try:
        with connection.cursor() as cursor:
            # Check if document_text_segments table has embedding field
            cursor.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'document_text_segments' 
                AND column_name = 'embedding';
            """)
            
            embedding_field = cursor.fetchone()
            if embedding_field:
                print(f"✅ Embedding field found: {embedding_field[0]} ({embedding_field[1]})")
                return True
            else:
                print("❌ No embedding field found in document_text_segments table")
                return False
                
    except Exception as e:
        print(f"❌ Error checking database schema: {e}")
        return False

def main():
    print("🚀 ReportMiner Setup Verification")
    print("=" * 50)
    
    # Check 1: pgvector
    pgvector_ok = check_pgvector()
    
    # Check 2: Existing data
    data_ok = check_existing_data()
    
    # Check 3: Database schema
    schema_ok = check_database_schema()
    
    print("\n" + "=" * 50)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 50)
    
    print(f"🔧 pgvector Extension: {'✅ Ready' if pgvector_ok else '❌ Issues'}")
    print(f"📊 Existing Data: {'✅ Ready' if data_ok else '⚠️ No embeddings'}")
    print(f"🗄️ Database Schema: {'✅ Ready' if schema_ok else '❌ Issues'}")
    
    if pgvector_ok and data_ok and schema_ok:
        print("\n🎉 READY FOR LANGCHAIN INTEGRATION!")
        print("Your setup is perfect for LangChain integration.")
        return True
    else:
        print("\n⚠️ SETUP NEEDS ATTENTION")
        if not pgvector_ok:
            print("   • Install pgvector extension")
        if not data_ok:
            print("   • Generate some embeddings first")
        if not schema_ok:
            print("   • Check database migration status")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
