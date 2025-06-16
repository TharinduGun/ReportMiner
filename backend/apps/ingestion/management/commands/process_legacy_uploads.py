from django.core.management.base import BaseCommand
from apps.ingestion.enhanced_upload_pipeline import BatchUploadPipeline
from apps.ingestion.models import FileUpload, Document


class Command(BaseCommand):
    help = 'Process existing FileUpload records through the enhanced pipeline'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--batch-size',
            type=int,
            default=10,
            help='Number of files to process in each batch'
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force reprocessing of already processed files'
        )
        parser.add_argument(
            '--file-type',
            type=str,
            choices=['pdf', 'docx', 'xlsx'],
            help='Process only files of specific type'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be processed without actually processing'
        )
    
    def handle(self, *args, **options):
        batch_size = options['batch_size']
        force = options['force']
        file_type = options.get('file_type')
        dry_run = options['dry_run']
        
        self.stdout.write(
            self.style.SUCCESS('Starting legacy upload processing...')
        )
        
        # Get unprocessed uploads
        if force:
            unprocessed_uploads = FileUpload.objects.all()
            if file_type:
                unprocessed_uploads = unprocessed_uploads.filter(type=file_type)
            self.stdout.write(f'Force mode: Processing all {unprocessed_uploads.count()} uploads')
        else:
            processed_filenames = set(Document.objects.values_list('filename', flat=True))
            unprocessed_uploads = FileUpload.objects.exclude(filename__in=processed_filenames)
            if file_type:
                unprocessed_uploads = unprocessed_uploads.filter(type=file_type)
            self.stdout.write(f'Found {unprocessed_uploads.count()} unprocessed uploads')
        
        if not unprocessed_uploads.exists():
            self.stdout.write(
                self.style.WARNING('No unprocessed uploads found.')
            )
            return
        
        # Show what would be processed in dry-run mode
        if dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN MODE - No actual processing'))
            for upload in unprocessed_uploads:
                self.stdout.write(f'  Would process: {upload.filename} ({upload.type})')
            return
        
        # Process in batches
        total_processed = 0
        total_successful = 0
        total_failed = 0
        
        pipeline = BatchUploadPipeline()
        
        for i in range(0, unprocessed_uploads.count(), batch_size):
            batch = list(unprocessed_uploads[i:i + batch_size])
            
            self.stdout.write(f'Processing batch {i//batch_size + 1}: {len(batch)} files')
            
            try:
                results = pipeline.process_batch(batch)
                
                total_processed += results['total_files']
                total_successful += results['successful']
                total_failed += results['failed']
                
                self.stdout.write(
                    f'Batch results: {results["successful"]} successful, {results["failed"]} failed'
                )
                
                # Show individual results
                for result in results['results']:
                    if result['success']:
                        self.stdout.write(
                            self.style.SUCCESS(f'  ✓ {result["filename"]} -> Document ID: {result["document_id"]}')
                        )
                    else:
                        self.stdout.write(
                            self.style.ERROR(f'  ✗ {result["filename"]}: {", ".join(result["errors"])}')
                        )
                
                if results['errors']:
                    for error in results['errors']:
                        self.stdout.write(
                            self.style.ERROR(f'Batch error: {error}')
                        )
                        
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'Batch processing failed: {str(e)}')
                )
                total_failed += len(batch)
        
        # Final summary
        self.stdout.write(
            self.style.SUCCESS(
                f'\n=== PROCESSING COMPLETED ===\n'
                f'Total processed: {total_processed}\n'
                f'Successful: {total_successful}\n'
                f'Failed: {total_failed}\n'
                f'Success rate: {(total_successful/total_processed)*100:.1f}%' if total_processed > 0 else 'Success rate: N/A'
            )
        )
        
        if total_failed > 0:
            self.stdout.write(
                self.style.WARNING(
                    f'\n{total_failed} files failed processing. '
                    f'Check the logs for details or use --force to retry.'
                )
            )