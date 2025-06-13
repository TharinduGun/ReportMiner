import uuid
from django.db import models

class Document(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    filename = models.CharField(max_length=255)
    original_filename = models.CharField(max_length=255)
    file_path = models.TextField(blank=True, null=True)
    file_type = models.CharField(max_length=10)
    file_size = models.BigIntegerField(blank=True, null=True)
    mime_type = models.CharField(max_length=100, blank=True, null=True)
    
    # Processing status
    processing_status = models.CharField(max_length=20, default='pending')
    processing_started_at = models.DateTimeField(blank=True, null=True)
    processing_completed_at = models.DateTimeField(blank=True, null=True)
    processing_error = models.TextField(blank=True, null=True)
    
    # Metadata (text fields for now, will upgrade to JSON later)
    metadata = models.TextField(default='{}', blank=True)
    extraction_summary = models.TextField(default='{}', blank=True)
    
    # Classification
    document_type = models.CharField(max_length=50, blank=True, null=True)
    language = models.CharField(max_length=10, default='en')
    page_count = models.IntegerField(blank=True, null=True)
    
    # Audit
    uploaded_by = models.CharField(max_length=100, blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        managed = True
        db_table = 'documents'
        
    def __str__(self):
        return self.filename

class DocumentTextSegment(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, db_column='document_id', related_name='text_segments')
    sequence_number = models.IntegerField()
    page_number = models.IntegerField(blank=True, null=True)
    section_id = models.CharField(max_length=100, blank=True, null=True)
    section_title = models.CharField(max_length=500, blank=True, null=True)
    content = models.TextField()
    content_cleaned = models.TextField(blank=True, null=True)
    content_length = models.IntegerField(blank=True, null=True)
    word_count = models.IntegerField(blank=True, null=True)
    segment_type = models.CharField(max_length=50, default='paragraph')
    
    # AI fields (text storage for now)
    embedding_text = models.TextField(blank=True, null=True)
    embedding_model = models.CharField(max_length=50, default='text-embedding-ada-002')
    extracted_entities = models.TextField(default='{}', blank=True)
    
    # Position (text storage for JSON)
    bbox_json = models.TextField(blank=True, null=True)
    font_info_json = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        managed = True
        db_table = 'document_text_segments'
        unique_together = (('document', 'sequence_number'),)
        
    def __str__(self):
        return f"{self.document.filename} - Segment {self.sequence_number}"

class DocumentTable(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, db_column='document_id', related_name='tables')
    table_name = models.CharField(max_length=255, blank=True, null=True)
    table_index = models.IntegerField(blank=True, null=True)
    sheet_name = models.CharField(max_length=255, blank=True, null=True)
    page_number = models.IntegerField(blank=True, null=True)
    row_count = models.IntegerField(blank=True, null=True)
    column_count = models.IntegerField(blank=True, null=True)
    has_header = models.BooleanField(default=True)
    
    # JSON as text fields
    table_data_json = models.TextField()
    column_definitions_json = models.TextField(blank=True, null=True)
    table_summary_json = models.TextField(default='{}', blank=True)
    bbox_json = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        managed = True
        db_table = 'document_tables'
        
    def __str__(self):
        return f"{self.document.filename} - Table {self.table_name or self.table_index}"

class DocumentStructuredData(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, db_column='document_id', related_name='structured_data')
    table = models.ForeignKey(DocumentTable, on_delete=models.CASCADE, db_column='table_id', related_name='cells', blank=True, null=True)
    row_number = models.IntegerField()
    column_number = models.IntegerField()
    column_name = models.CharField(max_length=255, blank=True, null=True)
    cell_value = models.TextField(blank=True, null=True)
    display_value = models.TextField(blank=True, null=True)
    text_value = models.TextField(blank=True, null=True)
    numeric_value = models.DecimalField(max_digits=20, decimal_places=6, blank=True, null=True)
    integer_value = models.BigIntegerField(blank=True, null=True)
    date_value = models.DateField(blank=True, null=True)
    datetime_value = models.DateTimeField(blank=True, null=True)
    boolean_value = models.BooleanField(blank=True, null=True)
    data_type = models.CharField(max_length=20, default='text')
    cell_metadata_json = models.TextField(default='{}', blank=True)
    confidence_score = models.DecimalField(max_digits=3, decimal_places=2, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        managed = True
        db_table = 'document_structured_data'
        
    def __str__(self):
        return f"{self.document.filename} - {self.column_name}: {self.cell_value}"

class DocumentKeyValue(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, db_column='document_id', related_name='key_values')
    key_name = models.CharField(max_length=255)
    key_category = models.CharField(max_length=100, blank=True, null=True)
    value_text = models.TextField(blank=True, null=True)
    value_numeric = models.DecimalField(max_digits=20, decimal_places=6, blank=True, null=True)
    value_date = models.DateField(blank=True, null=True)
    value_boolean = models.BooleanField(blank=True, null=True)
    page_number = models.IntegerField(blank=True, null=True)
    section_title = models.CharField(max_length=255, blank=True, null=True)
    extraction_method = models.CharField(max_length=50, blank=True, null=True)
    confidence_score = models.DecimalField(max_digits=3, decimal_places=2, blank=True, null=True)
    is_verified = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        managed = True
        db_table = 'document_key_values'
        
    def __str__(self):
        return f"{self.document.filename} - {self.key_name}: {self.value_text}"

# model for uploading files 
class FileUpload(models.Model): #choice for file types 
    FILE_TYPES = [
        ('pdf', 'PDF'),
        ('docx', 'Word Document'),
        ('xlsx', 'Excel Sheet'),
    ]

    filename = models.CharField(max_length=255) #File name
    file = models.FileField(upload_to='uploads/')# The actual file, stored in the 'uploads/' subdirectory inside /media
    type = models.CharField(max_length=10, choices=FILE_TYPES) #type of file
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):   # String representation of the object (for admin panel and logs)
        return self.filename