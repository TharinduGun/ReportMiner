#!/usr/bin/env python
"""
Quick fix script for pgvector issues
This temporarily replaces files to allow testing without pgvector
"""
import os
import shutil

def backup_and_replace_files():
    """Backup original files and replace with test versions"""
    files_to_replace = [
        ('models.py', 'models_test.py'),
        ('vector_processor.py', 'vector_processor_test.py')
    ]
    
    print("🔧 Applying temporary fixes for pgvector issues...")
    
    for original, test_version in files_to_replace:
        original_path = f"apps/ingestion/{original}"
        test_path = f"apps/ingestion/{test_version}"
        backup_path = f"apps/ingestion/{original}.backup"
        
        # Create backup
        if os.path.exists(original_path):
            if not os.path.exists(backup_path):
                shutil.copy2(original_path, backup_path)
                print(f"✅ Backed up {original} to {original}.backup")
            
            # Replace with test version
            if os.path.exists(test_path):
                shutil.copy2(test_path, original_path)
                print(f"✅ Replaced {original} with test version")
            else:
                print(f"❌ Test version {test_version} not found")
        else:
            print(f"❌ Original file {original} not found")

def restore_files():
    """Restore original files from backup"""
    files_to_restore = ['models.py', 'vector_processor.py']
    
    print("🔄 Restoring original files...")
    
    for filename in files_to_restore:
        original_path = f"apps/ingestion/{filename}"
        backup_path = f"apps/ingestion/{filename}.backup"
        
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, original_path)
            print(f"✅ Restored {filename} from backup")
        else:
            print(f"⚠️ No backup found for {filename}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "restore":
        restore_files()
    else:
        backup_and_replace_files()
        print("\n🧪 Files ready for testing! Run your tests now.")
        print("💡 When done, run: python quick_fix.py restore")
