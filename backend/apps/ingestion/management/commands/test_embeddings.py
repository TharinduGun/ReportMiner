"""
Management command to run embedding tests
"""
from django.core.management.base import BaseCommand
from django.test.utils import get_runner
from django.conf import settings
from django.test import TestCase
import sys
import unittest


class Command(BaseCommand):
    help = 'Run embedding-specific tests with detailed output'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--test-type',
            type=str,
            choices=['all', 'basic', 'performance', 'search', 'integration'],
            default='all',
            help='Type of tests to run'
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Run tests with verbose output'
        )
        parser.add_argument(
            '--fast',
            action='store_true',
            help='Skip slow performance tests'
        )
        parser.add_argument(
            '--coverage',
            action='store_true',
            help='Run with coverage report'
        )
    
    def handle(self, *args, **options):
        test_type = options['test_type']
        verbose = options['verbose']
        fast = options['fast']
        coverage = options['coverage']
        
        self.stdout.write(
            self.style.SUCCESS('🧪 RUNNING EMBEDDING TESTS')
        )
        self.stdout.write('=' * 50)
        
        # Determine which tests to run
        test_modules = self._get_test_modules(test_type, fast)
        
        if coverage:
            self._run_with_coverage(test_modules, verbose)
        else:
            self._run_tests(test_modules, verbose)
    
    def _get_test_modules(self, test_type, fast):
        """Get list of test modules to run based on type"""
        base_path = 'apps.ingestion.tests'
        
        if test_type == 'basic':
            return [f'{base_path}.test_embeddings.EmbeddingFilteringTestCase',
                   f'{base_path}.test_embeddings.EmbeddingGenerationTestCase']
        
        elif test_type == 'performance':
            if fast:
                return [f'{base_path}.test_performance.EmbeddingPerformanceTestCase.test_batch_processing_performance']
            else:
                return [f'{base_path}.test_performance']
        
        elif test_type == 'search':
            return [f'{base_path}.test_search']
        
        elif test_type == 'integration':
            return [f'{base_path}.test_embeddings.IntegrationTestCase',
                   f'{base_path}.test_search.SearchIntegrationTestCase']
        
        else:  # all
            modules = [f'{base_path}.test_embeddings']
            if not fast:
                modules.append(f'{base_path}.test_performance')
            modules.extend([
                f'{base_path}.test_search'
            ])
            return modules
    
    def _run_tests(self, test_modules, verbose):
        """Run tests using Django's test runner"""
        TestRunner = get_runner(settings)
        test_runner = TestRunner(verbosity=2 if verbose else 1, interactive=False)
        
        try:
            failures = test_runner.run_tests(test_modules)
            
            if failures:
                self.stdout.write(
                    self.style.ERROR(f'❌ {failures} test(s) failed')
                )
                sys.exit(1)
            else:
                self.stdout.write(
                    self.style.SUCCESS('✅ All tests passed!')
                )
                
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Test execution failed: {str(e)}')
            )
            sys.exit(1)
    
    def _run_with_coverage(self, test_modules, verbose):
        """Run tests with coverage report"""
        try:
            import coverage
        except ImportError:
            self.stdout.write(
                self.style.ERROR('❌ Coverage.py not installed. Install with: pip install coverage')
            )
            return
        
        # Start coverage
        cov = coverage.Coverage(source=['apps.ingestion'])
        cov.start()
        
        try:
            # Run tests
            self._run_tests(test_modules, verbose)
            
            # Stop coverage and generate report
            cov.stop()
            cov.save()
            
            self.stdout.write('\n' + '=' * 50)
            self.stdout.write(self.style.SUCCESS('📊 COVERAGE REPORT'))
            self.stdout.write('=' * 50)
            
            # Generate coverage report
            cov.report(show_missing=True)
            
            # Generate HTML report
            try:
                cov.html_report(directory='htmlcov')
                self.stdout.write(
                    self.style.SUCCESS('\n📁 HTML coverage report generated in htmlcov/ directory')
                )
            except Exception as e:
                self.stdout.write(
                    self.style.WARNING(f'⚠️ Could not generate HTML report: {str(e)}')
                )
                
        except Exception as e:
            cov.stop()
            self.stdout.write(
                self.style.ERROR(f'❌ Coverage test failed: {str(e)}')
            )


class EmbeddingTestSuite:
    """Test suite for embedding functionality"""
    
    @staticmethod
    def run_quick_tests():
        """Run essential tests quickly"""
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Add essential test cases
        from apps.ingestion.tests.test_embeddings import EmbeddingFilteringTestCase, EmbeddingGenerationTestCase
        
        suite.addTests(loader.loadTestsFromTestCase(EmbeddingFilteringTestCase))
        suite.addTests(loader.loadTestsFromTestCase(EmbeddingGenerationTestCase))
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
    
    @staticmethod
    def run_performance_tests():
        """Run performance-specific tests"""
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        from apps.ingestion.tests.test_performance import EmbeddingPerformanceTestCase
        
        suite.addTests(loader.loadTestsFromTestCase(EmbeddingPerformanceTestCase))
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
    
    @staticmethod
    def run_integration_tests():
        """Run integration tests"""
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        from apps.ingestion.tests.test_embeddings import IntegrationTestCase
        from apps.ingestion.tests.test_search import SearchIntegrationTestCase
        
        suite.addTests(loader.loadTestsFromTestCase(IntegrationTestCase))
        suite.addTests(loader.loadTestsFromTestCase(SearchIntegrationTestCase))
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
