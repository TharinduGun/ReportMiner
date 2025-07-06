#!/usr/bin/env python
"""
Quick test runner for embedding functionality
Run this script to test your embedding implementation
"""
import os
import sys
import django

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'reportminer.settings')
django.setup()

from apps.ingestion.management.commands.test_embeddings import EmbeddingTestSuite


def main():
    """Run embedding tests"""
    print("🧪 ReportMiner Embedding Test Suite")
    print("=" * 50)
    
    # Run quick tests first
    print("\n1️⃣ Running Essential Tests...")
    if EmbeddingTestSuite.run_quick_tests():
        print("✅ Essential tests passed!")
    else:
        print("❌ Essential tests failed!")
        return False
    
    # Ask user if they want to run performance tests
    run_performance = input("\n2️⃣ Run performance tests? (y/N): ").lower().startswith('y')
    
    if run_performance:
        print("\n⚡ Running Performance Tests...")
        if EmbeddingTestSuite.run_performance_tests():
            print("✅ Performance tests passed!")
        else:
            print("❌ Performance tests failed!")
    
    # Run integration tests
    print("\n3️⃣ Running Integration Tests...")
    if EmbeddingTestSuite.run_integration_tests():
        print("✅ Integration tests passed!")
    else:
        print("❌ Integration tests failed!")
        return False
    
    print("\n🎉 All tests completed successfully!")
    print("\n📊 Next steps:")
    print("  • Test with real documents: python manage.py test_pipeline --create-sample")
    print("  • Check embedding status: python manage.py embedding_status")
    print("  • Run full test suite: python manage.py test apps.ingestion")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
