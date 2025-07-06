"""
Quick script to check pgvector installation
"""
import os
import sys
import django
from django.db import connection

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'reportminer.settings')
django.setup()

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
                    return False
            else:
                print("❌ pgvector extension is not available")
                return False
                
    except Exception as e:
        print(f"❌ Error checking pgvector: {e}")
        return False

def install_pgvector():
    print("\n🔧 Attempting to install pgvector extension...")
    
    try:
        with connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            print("✅ pgvector extension installed successfully")
            return True
    except Exception as e:
        print(f"❌ Failed to install pgvector: {e}")
        return False

if __name__ == "__main__":
    if check_pgvector():
        print("🎉 pgvector is ready to use!")
    else:
        if install_pgvector():
            print("🎉 pgvector installation complete!")
        else:
            print("\n💡 Manual installation required:")
            print("1. Install pgvector on your PostgreSQL server")
            print("2. Connect to your database and run: CREATE EXTENSION vector;")
