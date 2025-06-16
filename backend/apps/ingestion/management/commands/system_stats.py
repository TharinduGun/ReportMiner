from django.core.management.base import BaseCommand
from django.db.models import Count, Avg, Sum, Q
from apps.ingestion.enhanced_upload_pipeline import get_processing_statistics
from apps.ingestion.models import Document, DocumentTextSegment, DocumentKeyValue, DocumentTable
import json
from datetime import datetime, timedelta


class Command(BaseCommand):
    help = 'Display system processing statistics'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--detailed',
            action='store_true',
            help='Show detailed statistics including recent documents and errors'
        )
        parser.add_argument(
            '--export-json',
            type=str,
            help='Export statistics to JSON file'
        )
        parser.add_argument(
            '--performance',
            action='store_true',
            help='Show performance metrics and processing times'
        )
        parser.add_argument(
            '--errors-only',
            action='store_true',
            help='Show only failed documents and error details'
        )
        parser.add_argument(
            '--days',
            type=int,
            default=30,
            help='Number of days to include in recent statistics (default: 30)'
        )
    
    def handle(self, *args, **options):
        detailed = options['detailed']
        export_file = options.get('export_json')
        performance = options['performance']
        errors_only = options['errors_only']
        days = options['days']
        
        # Get basic statistics
        stats = get_processing_statistics()
        
        if errors_only:
            self._show_errors_only()
            return
        
        # Display header
        self.stdout.write(
            self.style.SUCCESS('=' * 50)
        )
        self.stdout.write(
            self.style.SUCCESS('🚀 REPORTMINER SYSTEM STATISTICS')
        )
        self.stdout.write(
            self.style.SUCCESS('=' * 50)
        )
        
        # Documents overview
        doc_stats = stats['documents']
        self.stdout.write('\n📄 DOCUMENTS OVERVIEW:')
        self.stdout.write(f'  Total Documents: {doc_stats["total"]:,}')
        
        if doc_stats['by_status']:
            self.stdout.write('\n  📊 By Processing Status:')
            for status, count in doc_stats['by_status'].items():
                icon = self._get_status_icon(status)
                self.stdout.write(f'    {icon} {status.title()}: {count:,}')
        
        if doc_stats['by_type']:
            self.stdout.write('\n  📁 By File Type:')
            for file_type, count in doc_stats['by_type'].items():
                icon = self._get_file_type_icon(file_type)
                self.stdout.write(f'    {icon} {file_type.upper()}: {count:,}')
        
        # Content statistics
        content_stats = stats['content']
        self.stdout.write('\n📊 EXTRACTED CONTENT:')
        self.stdout.write(f'  📝 Text Segments: {content_stats["total_text_segments"]:,}')
        self.stdout.write(f'  📋 Tables: {content_stats["total_tables"]:,}')
        self.stdout.write(f'  🔑 Key-Value Pairs: {content_stats["total_key_values"]:,}')
        self.stdout.write(f'  🔢 Structured Cells: {content_stats["total_structured_cells"]:,}')
        
        # Recent activity
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_docs = Document.objects.filter(uploaded_at__gte=cutoff_date)
        
        if recent_docs.exists():
            self.stdout.write(f'\n📅 RECENT ACTIVITY (Last {days} days):')
            self.stdout.write(f'  📤 New Documents: {recent_docs.count():,}')
            
            recent_by_status = recent_docs.values('processing_status').annotate(
                count=Count('id')
            ).order_by('-count')
            
            for item in recent_by_status:
                icon = self._get_status_icon(item['processing_status'])
                self.stdout.write(f'    {icon} {item["processing_status"].title()}: {item["count"]:,}')
        
        # Performance metrics
        if performance:
            self._show_performance_metrics()
        
        # Detailed information
        if detailed:
            self._show_detailed_stats(days)
        
        # Export to JSON if requested
        if export_file:
            self._export_to_json(stats, export_file, days)
    
    def _show_performance_metrics(self):
        """Show performance and processing metrics"""
        self.stdout.write('\n⚡ PERFORMANCE METRICS:')
        
        # Average content per document
        avg_segments = DocumentTextSegment.objects.aggregate(
            avg_per_doc=Avg('sequence_number')
        )['avg_per_doc'] or 0
        
        avg_tables = DocumentTable.objects.values('document').annotate(
            table_count=Count('id')
        ).aggregate(avg_tables=Avg('table_count'))['avg_tables'] or 0
        
        avg_kv = DocumentKeyValue.objects.values('document').annotate(
            kv_count=Count('id')
        ).aggregate(avg_kv=Avg('kv_count'))['avg_kv'] or 0
        
        self.stdout.write(f'  📊 Avg Segments per Document: {avg_segments:.1f}')
        self.stdout.write(f'  📋 Avg Tables per Document: {avg_tables:.1f}')
        self.stdout.write(f'  🔑 Avg Key-Values per Document: {avg_kv:.1f}')
        
        # Document size distribution
        size_stats = Document.objects.exclude(file_size__isnull=True).aggregate(
            avg_size=Avg('file_size'),
            total_size=Sum('file_size')
        )
        
        if size_stats['avg_size']:
            avg_size_mb = size_stats['avg_size'] / (1024 * 1024)
            total_size_mb = size_stats['total_size'] / (1024 * 1024)
            self.stdout.write(f'  💾 Avg File Size: {avg_size_mb:.1f} MB')
            self.stdout.write(f'  💾 Total Storage: {total_size_mb:.1f} MB')
        
        # Processing success rate
        total_docs = Document.objects.count()
        completed_docs = Document.objects.filter(processing_status='completed').count()
        if total_docs > 0:
            success_rate = (completed_docs / total_docs) * 100
            self.stdout.write(f'  ✅ Processing Success Rate: {success_rate:.1f}%')
    
    def _show_detailed_stats(self, days):
        """Show detailed statistics"""
        self.stdout.write('\n' + '=' * 30)
        self.stdout.write('📋 DETAILED STATISTICS')
        self.stdout.write('=' * 30)
        
        # Recent documents
        recent_docs = Document.objects.order_by('-uploaded_at')[:10]
        if recent_docs:
            self.stdout.write('\n📅 RECENT DOCUMENTS (Last 10):')
            for doc in recent_docs:
                icon = self._get_status_icon(doc.processing_status)
                file_icon = self._get_file_type_icon(doc.file_type)
                upload_time = doc.uploaded_at.strftime('%Y-%m-%d %H:%M')
                
                # Get content counts
                segments = doc.text_segments.count()
                tables = doc.tables.count()
                kv_pairs = doc.key_values.count()
                
                self.stdout.write(
                    f'  {file_icon} {doc.filename[:40]:<40} {icon} '
                    f'[S:{segments} T:{tables} K:{kv_pairs}] {upload_time}'
                )
        
        # Document type distribution
        doc_types = Document.objects.exclude(document_type__isnull=True).values(
            'document_type'
        ).annotate(count=Count('id')).order_by('-count')
        
        if doc_types:
            self.stdout.write('\n📑 DOCUMENT TYPES:')
            for doc_type in doc_types:
                self.stdout.write(f'  📄 {doc_type["document_type"]}: {doc_type["count"]:,}')
        
        # Error analysis
        self._show_error_analysis()
        
        # Content quality metrics
        self._show_content_quality()
    
    def _show_error_analysis(self):
        """Show error analysis"""
        failed_docs = Document.objects.filter(processing_status='failed')
        
        if failed_docs.exists():
            self.stdout.write(f'\n❌ ERROR ANALYSIS ({failed_docs.count()} failed documents):')
            
            # Group by error type
            error_summary = {}
            for doc in failed_docs[:20]:  # Limit to prevent overflow
                error = doc.processing_error or 'Unknown error'
                # Extract first line of error for grouping
                error_key = error.split('\n')[0][:100]
                if error_key not in error_summary:
                    error_summary[error_key] = []
                error_summary[error_key].append(doc.filename)
            
            for error, files in error_summary.items():
                self.stdout.write(f'  🔥 {error}')
                self.stdout.write(f'     Affected files ({len(files)}): {", ".join(files[:3])}{"..." if len(files) > 3 else ""}')
    
    def _show_content_quality(self):
        """Show content quality metrics"""
        self.stdout.write('\n📊 CONTENT QUALITY METRICS:')
        
        # Documents with no extracted content
        empty_docs = Document.objects.filter(
            processing_status='completed',
            text_segments__isnull=True
        ).distinct().count()
        
        # Documents with very few segments (potentially poor extraction)
        low_content_docs = Document.objects.filter(
            processing_status='completed'
        ).annotate(
            segment_count=Count('text_segments')
        ).filter(segment_count__lt=3).count()
        
        # Documents with tables
        docs_with_tables = Document.objects.filter(
            tables__isnull=False
        ).distinct().count()
        
        # Documents with key-value pairs
        docs_with_kv = Document.objects.filter(
            key_values__isnull=False
        ).distinct().count()
        
        total_completed = Document.objects.filter(processing_status='completed').count()
        
        if total_completed > 0:
            self.stdout.write(f'  📝 Documents with no content: {empty_docs} ({(empty_docs/total_completed)*100:.1f}%)')
            self.stdout.write(f'  ⚠️  Documents with low content: {low_content_docs} ({(low_content_docs/total_completed)*100:.1f}%)')
            self.stdout.write(f'  📋 Documents with tables: {docs_with_tables} ({(docs_with_tables/total_completed)*100:.1f}%)')
            self.stdout.write(f'  🔑 Documents with key-values: {docs_with_kv} ({(docs_with_kv/total_completed)*100:.1f}%)')
    
    def _show_errors_only(self):
        """Show only error information"""
        self.stdout.write(
            self.style.ERROR('❌ FAILED DOCUMENTS REPORT')
        )
        self.stdout.write('=' * 40)
        
        failed_docs = Document.objects.filter(processing_status='failed').order_by('-uploaded_at')
        
        if not failed_docs.exists():
            self.stdout.write(
                self.style.SUCCESS('🎉 No failed documents found!')
            )
            return
        
        self.stdout.write(f'Total failed documents: {failed_docs.count()}')
        
        for doc in failed_docs:
            self.stdout.write(f'\n📄 {doc.filename}')
            self.stdout.write(f'   File Type: {doc.file_type}')
            self.stdout.write(f'   Upload Date: {doc.uploaded_at.strftime("%Y-%m-%d %H:%M:%S")}')
            self.stdout.write(f'   Error: {doc.processing_error or "No error message"}')
            
            if doc.file_size:
                size_mb = doc.file_size / (1024 * 1024)
                self.stdout.write(f'   File Size: {size_mb:.2f} MB')
    
    def _export_to_json(self, stats, export_file, days):
        """Export statistics to JSON file"""
        try:
            # Add additional data for export
            export_data = {
                'generated_at': datetime.now().isoformat(),
                'report_period_days': days,
                'basic_stats': stats
            }
            
            # Add recent activity
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_docs = Document.objects.filter(uploaded_at__gte=cutoff_date)
            
            export_data['recent_activity'] = {
                'period_days': days,
                'new_documents': recent_docs.count(),
                'by_status': dict(recent_docs.values('processing_status').annotate(
                    count=Count('id')
                ).values_list('processing_status', 'count'))
            }
            
            # Add error information
            failed_docs = Document.objects.filter(processing_status='failed')
            export_data['errors'] = {
                'total_failed': failed_docs.count(),
                'recent_errors': [
                    {
                        'filename': doc.filename,
                        'error': doc.processing_error,
                        'upload_date': doc.uploaded_at.isoformat()
                    }
                    for doc in failed_docs.order_by('-uploaded_at')[:10]
                ]
            }
            
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.stdout.write(
                self.style.SUCCESS(f'\n📁 Statistics exported to: {export_file}')
            )
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Failed to export statistics: {str(e)}')
            )
    
    def _get_status_icon(self, status):
        """Get icon for processing status"""
        icons = {
            'completed': '✅',
            'failed': '❌',
            'pending': '⏳',
            'processing': '⚙️',
            'requires_review': '⚠️'
        }
        return icons.get(status, '❓')
    
    def _get_file_type_icon(self, file_type):
        """Get icon for file type"""
        icons = {
            'pdf': '📕',
            'docx': '📘',
            'xlsx': '📗',
            'csv': '📊',
            'txt': '📄'
        }
        return icons.get(file_type, '📄')