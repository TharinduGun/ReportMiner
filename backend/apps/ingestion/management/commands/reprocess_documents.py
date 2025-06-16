from django.core.management.base import BaseCommand
from apps.ingestion.models import Document, FileUpload
from apps.ingestion.enhanced_upload_pipeline import EnhancedUploadPipeline
import json


class Command(BaseCommand):
    help = 'Reprocess existing documents through the enhanced pipeline'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--document-id',
            type=str,
            help='Specific document ID to reprocess'
        )
        parser.add_argument(
            '--status',
            type=str,
            choices=['failed', 'pending', 'processing', 'completed'],
            help='Reprocess documents with specific status'
        )
        parser.add_argument(
            '--file-type',
            type=str,
            choices=['pdf', 'docx', 'xlsx'],
            help='Reprocess documents of specific file type'
        )
        parser.add_argument(
            '--document-type',
            type=str,
            help='Reprocess documents of specific document type'
        )
        parser.add_argument(
            '--clear-data',
            action='store_true',
            help='Clear existing processed data before reprocessing'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be reprocessed without actually doing it'
        )
        parser.add_argument(
            '--limit',
            type=int,
            default=100,
            help='Maximum number of documents to process (default: 100)'
        )
    
    def handle(self, *args, **options):
        document_id = options.get('document_id')
        status_filter = options.get('status')
        file_type = options.get('file_type')
        document_type = options.get('document_type')
        clear_data = options['clear_data']
        dry_run = options['dry_run']
        limit = options['limit']
        
        # Build queryset
        if document_id:
            try:
                documents = Document.objects.filter(id=document_id)
                if not documents.exists():
                    self.stdout.write(
                        self.style.ERROR(f'Document with ID {document_id} not found')
                    )
                    return
            except ValueError:
                self.stdout.write(
                    self.style.ERROR('Invalid document ID format')
                )
                return
        else:
            documents = Document.objects.all()
            
            if status_filter:
                documents = documents.filter(processing_status=status_filter)
            if file_type:
                documents = documents.filter(file_type=file_type)
            if document_type:
                documents = documents.filter(document_type=document_type)
            
            # Apply limit
            documents = documents.order_by('-uploaded_at')[:limit]
        
        if not documents.exists():
            self.stdout.write(
                self.style.WARNING('No documents found matching criteria')
            )
            return
        
        self.stdout.write(f'Found {documents.count()} documents to reprocess')
        
        # Show summary of what will be processed
        if dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN MODE - No actual reprocessing'))
            for doc in documents:
                segments_count = doc.text_segments.count()
                tables_count = doc.tables.count()
                kv_count = doc.key_values.count()
                self.stdout.write(
                    f'  Would reprocess: {doc.filename} '
                    f'(Status: {doc.processing_status}, '
                    f'Segments: {segments_count}, '
                    f'Tables: {tables_count}, '
                    f'KV pairs: {kv_count})'
                )
            return
        
        # Confirm before proceeding
        if clear_data and not document_id:
            confirm = input(
                f'This will delete existing data for {documents.count()} documents. '
                f'Are you sure? (yes/no): '
            )
            if confirm.lower() != 'yes':
                self.stdout.write('Operation cancelled.')
                return
        
        pipeline = EnhancedUploadPipeline()
        successful = 0
        failed = 0
        skipped = 0
        
        for document in documents:
            self.stdout.write(f'\nProcessing: {document.filename} (ID: {document.id})')
            
            try:
                # Find original FileUpload
                file_upload = None
                try:
                    metadata = json.loads(document.metadata) if document.metadata != '{}' else {}
                    legacy_id = metadata.get('legacy_upload_id')
                    if legacy_id:
                        file_upload = FileUpload.objects.get(id=legacy_id)
                except Exception as e:
                    self.stdout.write(f'  Warning: Could not load metadata: {str(e)}')
                
                if not file_upload:
                    self.stdout.write(
                        self.style.WARNING(f'  ⚠ No original file upload found for {document.filename}')
                    )
                    skipped += 1
                    continue
                
                # Show current data counts
                current_segments = document.text_segments.count()
                current_tables = document.tables.count()
                current_kv = document.key_values.count()
                current_cells = document.structured_data.count()
                
                self.stdout.write(
                    f'  Current data: {current_segments} segments, '
                    f'{current_tables} tables, {current_kv} KV pairs, '
                    f'{current_cells} cells'
                )
                
                # Clear existing data if requested
                if clear_data:
                    self.stdout.write('  Clearing existing data...')
                    document.text_segments.all().delete()
                    document.tables.all().delete()
                    document.key_values.all().delete()
                    document.structured_data.all().delete()
                
                # Reprocess
                self.stdout.write('  Extracting text...')
                raw_text = pipeline._extract_text(file_upload)
                
                self.stdout.write('  Processing content...')
                result = pipeline._process_document_content(document, raw_text)
                
                if result.get('processing_status') == 'completed':
                    successful += 1
                    self.stdout.write(
                        self.style.SUCCESS(
                            f'  ✓ Successfully reprocessed {document.filename}\n'
                            f'    Segments: {result.get("segments_created", 0)}, '
                            f'    Tables: {result.get("tables_detected", 0)}, '
                            f'    KV pairs: {result.get("key_values_extracted", 0)}'
                        )
                    )
                else:
                    failed += 1
                    errors = result.get('errors', ['Unknown error'])
                    self.stdout.write(
                        self.style.ERROR(
                            f'  ✗ Failed to reprocess {document.filename}: {"; ".join(errors)}'
                        )
                    )
                    
            except Exception as e:
                failed += 1
                self.stdout.write(
                    self.style.ERROR(f'  ✗ Error reprocessing {document.filename}: {str(e)}')
                )
        
        # Summary
        total = successful + failed + skipped
        self.stdout.write(
            self.style.SUCCESS(
                f'\n=== REPROCESSING COMPLETED ===\n'
                f'Total documents: {total}\n'
                f'Successful: {successful}\n'
                f'Failed: {failed}\n'
                f'Skipped: {skipped}\n'
                f'Success rate: {(successful/max(total-skipped, 1))*100:.1f}%'
            )
        )
        
        if failed > 0:
            self.stdout.write(
                self.style.WARNING(
                    f'\n{failed} documents failed reprocessing. '
                    f'Check the error messages above for details.'
                )
            )