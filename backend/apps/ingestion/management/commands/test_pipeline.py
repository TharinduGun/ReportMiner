from django.core.management.base import BaseCommand
from django.core.files.uploadedfile import SimpleUploadedFile
from apps.ingestion.models import FileUpload, Document
from apps.ingestion.enhanced_upload_pipeline import EnhancedUploadPipeline, DocumentSearchPipeline
import os
import tempfile
import json


class Command(BaseCommand):
    help = 'Test the enhanced upload pipeline with sample data'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--create-sample',
            action='store_true',
            help='Create sample test files for testing'
        )
        parser.add_argument(
            '--test-file',
            type=str,
            help='Path to specific file to test'
        )
        parser.add_argument(
            '--test-search',
            action='store_true',
            help='Test search functionality with sample queries'
        )
        parser.add_argument(
            '--test-existing',
            action='store_true',
            help='Test pipeline with existing FileUpload records'
        )
        parser.add_argument(
            '--cleanup',
            action='store_true',
            help='Clean up test data after testing'
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Show detailed output during testing'
        )
    
    def handle(self, *args, **options):
        create_sample = options['create_sample']
        test_file_path = options.get('test_file')
        test_search = options['test_search']
        test_existing = options['test_existing']
        cleanup = options['cleanup']
        self.verbose = options['verbose']
        
        self.stdout.write(
            self.style.SUCCESS('🧪 REPORTMINER PIPELINE TESTING')
        )
        self.stdout.write('=' * 50)
        
        test_results = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'created_documents': []
        }
        
        try:
            if create_sample:
                result = self.create_sample_files()
                test_results['tests_run'] += 1
                if result['success']:
                    test_results['tests_passed'] += 1
                    test_results['created_documents'].extend(result.get('document_ids', []))
                else:
                    test_results['tests_failed'] += 1
            
            if test_file_path:
                result = self.test_specific_file(test_file_path)
                test_results['tests_run'] += 1
                if result['success']:
                    test_results['tests_passed'] += 1
                    if result.get('document_id'):
                        test_results['created_documents'].append(result['document_id'])
                else:
                    test_results['tests_failed'] += 1
            
            if test_existing:
                result = self.test_existing_uploads()
                test_results['tests_run'] += 1
                if result['success']:
                    test_results['tests_passed'] += 1
                    test_results['created_documents'].extend(result.get('document_ids', []))
                else:
                    test_results['tests_failed'] += 1
            
            if test_search:
                result = self.test_search_functionality()
                test_results['tests_run'] += 1
                if result['success']:
                    test_results['tests_passed'] += 1
                else:
                    test_results['tests_failed'] += 1
            
            # If no specific test requested, run basic test
            if not any([create_sample, test_file_path, test_search, test_existing]):
                result = self.run_basic_test()
                test_results['tests_run'] += 1
                if result['success']:
                    test_results['tests_passed'] += 1
                    test_results['created_documents'].extend(result.get('document_ids', []))
                else:
                    test_results['tests_failed'] += 1
            
            # Cleanup if requested
            if cleanup and test_results['created_documents']:
                self.cleanup_test_data(test_results['created_documents'])
            
            # Final summary
            self.show_test_summary(test_results)
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Testing failed with error: {str(e)}')
            )
    
    def create_sample_files(self):
        """Create sample test files for testing"""
        self.stdout.write('\n📝 Creating sample test files...')
        
        sample_files = [
            self._create_sample_pdf(),
            self._create_sample_docx(),
            self._create_sample_xlsx()
        ]
        
        created_documents = []
        failed_files = []
        
        for sample_data in sample_files:
            try:
                result = self._process_sample_file(sample_data)
                if result['success']:
                    created_documents.append(result['document_id'])
                    self.stdout.write(
                        self.style.SUCCESS(
                            f'  ✅ Created and processed: {sample_data["filename"]} '
                            f'(ID: {result["document_id"]})'
                        )
                    )
                    
                    if self.verbose:
                        stats = result.get('processing_stats', {})
                        self.stdout.write(
                            f'     📊 Segments: {stats.get("segments_created", 0)}, '
                            f'Tables: {stats.get("tables_detected", 0)}, '
                            f'Key-Values: {stats.get("key_values_extracted", 0)}'
                        )
                else:
                    failed_files.append(sample_data['filename'])
                    self.stdout.write(
                        self.style.ERROR(f'  ❌ Failed to process: {sample_data["filename"]}')
                    )
                    
            except Exception as e:
                failed_files.append(sample_data['filename'])
                self.stdout.write(
                    self.style.ERROR(f'  ❌ Error creating {sample_data["filename"]}: {str(e)}')
                )
        
        success = len(failed_files) == 0
        
        if success:
            self.stdout.write(
                self.style.SUCCESS(f'\n✅ Successfully created {len(created_documents)} sample documents')
            )
        else:
            self.stdout.write(
                self.style.WARNING(f'\n⚠️ Created {len(created_documents)} documents, {len(failed_files)} failed')
            )
        
        return {
            'success': success,
            'document_ids': created_documents,
            'failed_files': failed_files
        }
    
    def _create_sample_pdf(self):
        """Create sample PDF content"""
        return {
            'filename': 'test_financial_report.pdf',
            'file_type': 'pdf',
            'content': """
QUARTERLY FINANCIAL REPORT
Q4 2023 Performance Summary

EXECUTIVE SUMMARY
This report provides a comprehensive overview of our financial performance for Q4 2023.
The company achieved significant growth across all key metrics.

KEY FINANCIAL METRICS
Revenue: $2,500,000
Operating Profit: $450,000
Net Profit: $380,000
Total Expenses: $2,120,000
Growth Rate: 15.3%
Market Share: 12.5%

QUARTERLY BREAKDOWN
Month    Revenue    Expenses    Net Profit
Oct      820,000    690,000     130,000
Nov      840,000    710,000     130,000
Dec      840,000    720,000     120,000
Total    2,500,000  2,120,000   380,000

DEPARTMENT PERFORMANCE
Sales Department: Exceeded targets by 12%
Marketing Department: ROI of 3.2x
Operations Department: Cost reduction of 8%
R&D Department: 3 new products launched

MARKET ANALYSIS
The technology sector showed strong growth in Q4 2023.
Customer satisfaction increased to 94%.
Employee retention rate: 89%.

FUTURE OUTLOOK
Looking ahead to 2024, we expect continued growth.
Projected revenue increase: 18-22%
New market expansion planned for Q2 2024.

RISK FACTORS
Supply chain disruptions
Increased competition
Regulatory changes
Currency fluctuations

CONCLUSION
Q4 2023 was a successful quarter with strong financial performance.
The company is well-positioned for continued growth in 2024.
            """
        }
    
    def _create_sample_docx(self):
        """Create sample DOCX content"""
        return {
            'filename': 'test_project_proposal.docx',
            'file_type': 'docx',
            'content': """
PROJECT PROPOSAL DOCUMENT
AI-Powered Document Analysis System

PROJECT OVERVIEW
Project Name: ReportMiner Enhancement
Project Manager: John Smith
Start Date: January 15, 2024
End Date: June 30, 2024
Total Budget: $150,000
Team Size: 5 members

PROJECT OBJECTIVES
Primary Goal: Develop advanced AI document processing capabilities
Secondary Goal: Improve extraction accuracy by 25%
Tertiary Goal: Reduce processing time by 40%

TEAM COMPOSITION
Role: Software Engineer
Name: Alice Johnson
Experience: 5 years
Specialization: Machine Learning

Role: Data Scientist  
Name: Bob Wilson
Experience: 3 years
Specialization: NLP

Role: QA Engineer
Name: Carol Davis
Experience: 4 years
Specialization: Test Automation

TECHNICAL REQUIREMENTS
Programming Languages: Python, JavaScript
Frameworks: Django, React
Database: PostgreSQL
AI Models: GPT-4, BERT
Cloud Platform: AWS

BUDGET BREAKDOWN
Personnel Costs: $120,000
Software Licenses: $15,000
Hardware: $10,000
Training: $3,000
Miscellaneous: $2,000

PROJECT PHASES
Phase 1: Requirements Analysis (2 weeks)
Phase 2: System Design (3 weeks)
Phase 3: Development (12 weeks)
Phase 4: Testing (3 weeks)
Phase 5: Deployment (2 weeks)

DELIVERABLES
Technical Specification Document
Working Prototype
User Documentation
Training Materials
Deployment Guide

RISK ASSESSMENT
Technical Risk: Medium
Timeline Risk: Low
Budget Risk: Low
Resource Risk: Medium

SUCCESS METRICS
Processing accuracy: >95%
User satisfaction: >90%
Performance improvement: >40%
Bug rate: <0.1%

CONCLUSION
This project will significantly enhance our document processing capabilities.
Expected ROI: 200% within first year.
            """
        }
    
    def _create_sample_xlsx(self):
        """Create sample XLSX content"""
        return {
            'filename': 'test_sales_data.xlsx',
            'file_type': 'xlsx',
            'content': """
Sheet: Monthly Sales Data

Month      Product A    Product B    Product C    Total
January    45,000      32,000       28,000      105,000
February   52,000      38,000       31,000      121,000
March      48,000      35,000       29,000      112,000
April      55,000      42,000       33,000      130,000
May        58,000      45,000       35,000      138,000
June       62,000      48,000       38,000      148,000
July       65,000      51,000       40,000      156,000
August     68,000      54,000       42,000      164,000
September  71,000      57,000       44,000      172,000
October    74,000      60,000       46,000      180,000
November   77,000      63,000       48,000      188,000
December   80,000      66,000       50,000      196,000
Total      715,000     591,000      464,000     1,770,000

Sheet: Summary Statistics

Metric                Value
Total Annual Revenue  1,770,000
Average Monthly Revenue  147,500
Best Performing Month    December
Best Product             Product A
Growth Rate              12.5%
Market Share             15.2%
Customer Count           2,500
Average Order Value      708
Return Rate              2.3%
Customer Satisfaction    4.6

Sheet: Regional Performance

Region      Q1         Q2         Q3         Q4         Total
North       95,000     125,000    145,000    165,000    530,000
South       85,000     110,000    135,000    155,000    485,000
East        75,000     100,000    120,000    140,000    435,000
West        65,000     85,000     105,000    125,000    380,000
Total       320,000    420,000    505,000    585,000    1,830,000
            """
        }
    
    def _process_sample_file(self, sample_data):
        """Process a sample file through the pipeline"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{sample_data["file_type"]}', delete=False) as f:
                f.write(sample_data['content'])
                temp_path = f.name
            
            try:
                # Create FileUpload record
                with open(temp_path, 'rb') as f:
                    file_upload = FileUpload.objects.create(
                        filename=sample_data['filename'],
                        file=SimpleUploadedFile(sample_data['filename'], f.read()),
                        type=sample_data['file_type']
                    )
                
                # Process through pipeline
                pipeline = EnhancedUploadPipeline()
                result = pipeline.process_uploaded_file(file_upload)
                
                return {
                    'success': result['success'],
                    'document_id': result.get('document_id'),
                    'processing_stats': result.get('processing_results', {}),
                    'errors': result.get('errors', [])
                }
                
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_specific_file(self, file_path):
        """Test the pipeline with a specific file"""
        self.stdout.write(f'\n📁 Testing specific file: {file_path}')
        
        if not os.path.exists(file_path):
            self.stdout.write(
                self.style.ERROR(f'❌ File not found: {file_path}')
            )
            return {'success': False, 'error': 'File not found'}
        
        # Determine file type
        ext = os.path.splitext(file_path)[1].lower()
        file_type_map = {'.pdf': 'pdf', '.docx': 'docx', '.xlsx': 'xlsx', '.csv': 'csv', '.txt': 'txt'}
        file_type = file_type_map.get(ext)
        
        if not file_type:
            self.stdout.write(
                self.style.ERROR(f'❌ Unsupported file type: {ext}')
            )
            return {'success': False, 'error': f'Unsupported file type: {ext}'}
        
        filename = os.path.basename(file_path)
        
        try:
            # Create FileUpload record
            with open(file_path, 'rb') as f:
                file_upload = FileUpload.objects.create(
                    filename=filename,
                    file=SimpleUploadedFile(filename, f.read()),
                    type=file_type
                )
            
            # Test pipeline
            result = self.test_pipeline_with_upload(file_upload)
            return result
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Failed to create file upload: {str(e)}')
            )
            return {'success': False, 'error': str(e)}
    
    def test_existing_uploads(self):
        """Test pipeline with existing FileUpload records"""
        self.stdout.write('\n📋 Testing with existing FileUpload records...')
        
        uploads = FileUpload.objects.all()[:3]  # Test first 3 uploads
        
        if not uploads:
            self.stdout.write(
                self.style.WARNING('⚠️ No FileUpload records found.')
            )
            return {'success': False, 'error': 'No FileUpload records found'}
        
        document_ids = []
        failed_count = 0
        
        for upload in uploads:
            result = self.test_pipeline_with_upload(upload)
            if result['success']:
                if result.get('document_id'):
                    document_ids.append(result['document_id'])
            else:
                failed_count += 1
        
        success = failed_count == 0
        
        if success:
            self.stdout.write(
                self.style.SUCCESS(f'✅ Successfully processed {len(uploads)} existing uploads')
            )
        else:
            self.stdout.write(
                self.style.WARNING(f'⚠️ Processed {len(uploads) - failed_count}/{len(uploads)} uploads')
            )
        
        return {
            'success': success,
            'document_ids': document_ids
        }
    
    def test_pipeline_with_upload(self, file_upload):
        """Test the enhanced pipeline with a FileUpload"""
        self.stdout.write(f'\n🔄 Processing: {file_upload.filename}')
        
        try:
            pipeline = EnhancedUploadPipeline()
            result = pipeline.process_uploaded_file(file_upload)
            
            if result['success']:
                self.stdout.write(
                    self.style.SUCCESS('  ✅ Pipeline processing successful!')
                )
                
                # Display results
                stats = result['processing_results']
                self.stdout.write(f'  📄 Document ID: {result["document_id"]}')
                self.stdout.write(f'  📝 Text segments: {stats.get("segments_created", 0)}')
                self.stdout.write(f'  📋 Tables detected: {stats.get("tables_detected", 0)}')
                self.stdout.write(f'  🔑 Key-values extracted: {stats.get("key_values_extracted", 0)}')
                
                # Show text preview
                if self.verbose:
                    preview = result.get('text_preview', '')
                    if preview:
                        self.stdout.write(f'  📖 Text preview: {preview[:200]}...')
                
                return {
                    'success': True,
                    'document_id': result['document_id'],
                    'stats': stats
                }
                
            else:
                self.stdout.write(
                    self.style.ERROR('  ❌ Pipeline processing failed!')
                )
                for error in result.get('errors', []):
                    self.stdout.write(f'    💥 Error: {error}')
                
                return {
                    'success': False,
                    'errors': result.get('errors', [])
                }
                    
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'  ❌ Pipeline test failed: {str(e)}')
            )
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_search_functionality(self):
        """Test search functionality"""
        self.stdout.write('\n🔍 Testing search functionality...')
        
        # Check if we have any documents
        if not Document.objects.filter(processing_status='completed').exists():
            self.stdout.write(
                self.style.WARNING('⚠️ No completed documents found for search testing')
            )
            return {'success': False, 'error': 'No documents to search'}
        
        # Test queries
        test_queries = [
            'revenue',
            'financial',
            'project',
            'sales',
            'performance'
        ]
        
        successful_searches = 0
        
        for query in test_queries:
            try:
                results = DocumentSearchPipeline.search_documents(query, limit=5)
                result_count = len(results['results'])
                
                self.stdout.write(f'  🔎 Query "{query}": {result_count} results')
                
                if self.verbose and result_count > 0:
                    for i, result in enumerate(results['results'][:2]):  # Show first 2
                        score = result['relevance_score']
                        filename = result['filename']
                        self.stdout.write(f'    {i+1}. {filename} (score: {score:.3f})')
                
                successful_searches += 1
                
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'  ❌ Search failed for "{query}": {str(e)}')
                )
        
        success = successful_searches == len(test_queries)
        
        if success:
            self.stdout.write(
                self.style.SUCCESS(f'✅ All {len(test_queries)} search queries completed successfully')
            )
        else:
            self.stdout.write(
                self.style.WARNING(f'⚠️ {successful_searches}/{len(test_queries)} searches completed')
            )
        
        return {'success': success}
    
    def run_basic_test(self):
        """Run basic test if no specific test is requested"""
        self.stdout.write('\n🧪 Running basic pipeline test...')
        
        # First check existing uploads
        existing_uploads = FileUpload.objects.all()[:1]
        
        if existing_uploads:
            self.stdout.write('📋 Testing with existing upload...')
            result = self.test_pipeline_with_upload(existing_uploads[0])
            if result['success']:
                return {
                    'success': True,
                    'document_ids': [result['document_id']] if result.get('document_id') else []
                }
        
        # If no existing uploads or test failed, create sample
        self.stdout.write('📝 Creating test sample...')
        sample_data = self._create_sample_pdf()
        result = self._process_sample_file(sample_data)
        
        if result['success']:
            self.stdout.write(
                self.style.SUCCESS('✅ Basic test completed successfully!')
            )
            return {
                'success': True,
                'document_ids': [result['document_id']] if result.get('document_id') else []
            }
        else:
            self.stdout.write(
                self.style.ERROR('❌ Basic test failed!')
            )
            return {'success': False}
    
    def cleanup_test_data(self, document_ids):
        """Clean up test data"""
        self.stdout.write(f'\n🧹 Cleaning up {len(document_ids)} test documents...')
        
        deleted_count = 0
        for doc_id in document_ids:
            try:
                document = Document.objects.get(id=doc_id)
                filename = document.filename
                
                # Delete the document (cascade will handle related data)
                document.delete()
                deleted_count += 1
                
                self.stdout.write(f'  🗑️ Deleted: {filename}')
                
            except Document.DoesNotExist:
                self.stdout.write(f'  ⚠️ Document {doc_id} not found')
            except Exception as e:
                self.stdout.write(f'  ❌ Error deleting {doc_id}: {str(e)}')
        
        self.stdout.write(
            self.style.SUCCESS(f'✅ Cleanup completed: {deleted_count} documents deleted')
        )
    
    def show_test_summary(self, results):
        """Show test summary"""
        self.stdout.write('\n' + '=' * 50)
        self.stdout.write(
            self.style.SUCCESS('📊 TEST SUMMARY')
        )
        self.stdout.write('=' * 50)
        
        self.stdout.write(f'Tests Run: {results["tests_run"]}')
        self.stdout.write(f'Tests Passed: {results["tests_passed"]}')
        self.stdout.write(f'Tests Failed: {results["tests_failed"]}')
        
        if results['tests_run'] > 0:
            success_rate = (results['tests_passed'] / results['tests_run']) * 100
            self.stdout.write(f'Success Rate: {success_rate:.1f}%')
        
        if results['created_documents']:
            self.stdout.write(f'Documents Created: {len(results["created_documents"])}')
            
            if self.verbose:
                self.stdout.write('\nCreated Document IDs:')
                for doc_id in results['created_documents']:
                    self.stdout.write(f'  📄 {doc_id}')
        
        # Overall status
        if results['tests_failed'] == 0 and results['tests_run'] > 0:
            self.stdout.write(
                self.style.SUCCESS('\n🎉 ALL TESTS PASSED!')
            )
        elif results['tests_passed'] > 0:
            self.stdout.write(
                self.style.WARNING('\n⚠️ SOME TESTS FAILED')
            )
        else:
            self.stdout.write(
                self.style.ERROR('\n❌ ALL TESTS FAILED')
            )
            