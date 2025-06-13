from django.contrib import admin
from .models import (
    Document, DocumentTextSegment, DocumentTable, 
    DocumentStructuredData, DocumentKeyValue, FileUpload
)

@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ['filename', 'file_type', 'processing_status', 'uploaded_at']
    list_filter = ['file_type', 'processing_status']
    search_fields = ['filename', 'original_filename']
    readonly_fields = ['id', 'created_at', 'updated_at']

@admin.register(DocumentTextSegment)
class DocumentTextSegmentAdmin(admin.ModelAdmin):
    list_display = ['id', 'sequence_number', 'segment_type', 'word_count']
    list_filter = ['segment_type']
    search_fields = ['content']

@admin.register(DocumentTable)
class DocumentTableAdmin(admin.ModelAdmin):
    list_display = ['id', 'table_name', 'row_count', 'column_count']

@admin.register(DocumentStructuredData)
class DocumentStructuredDataAdmin(admin.ModelAdmin):
    list_display = ['id', 'column_name', 'cell_value', 'data_type']
    list_filter = ['data_type']

@admin.register(DocumentKeyValue)
class DocumentKeyValueAdmin(admin.ModelAdmin):
    list_display = ['id', 'key_name', 'value_text', 'key_category']
    list_filter = ['key_category']

# Register your models here.
admin.site.register(FileUpload) # FileUpload model