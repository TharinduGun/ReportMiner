# backend/apps/ingestion/models.py

import uuid
from django.db import models


class FileUpload(models.Model):
    """
    (Re-added!) your original FileUpload model from before.
    e.g.:
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    # … any other fields …

    def __str__(self):
        return f"FileUpload {self.id} – {self.file.name}"


# Document model to track ingestion pipeline status and metrics
class Document(models.Model):
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('RUNNING', 'Running'),
        ('SUCCESS', 'Success'),
        ('ERROR', 'Error'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    file = models.FileField(upload_to='documents/%Y/%m/%d/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    # New fields to record pipeline metrics
    chunk_count = models.IntegerField(null=True, blank=True)
    total_tokens = models.IntegerField(null=True, blank=True)

    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='PENDING')
    error_message = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"Document {self.id} – {self.file.name}"

    # ----- status‐update helpers -----

    def mark_processing(self):
        """Called at the start of processing."""
        self.status = 'RUNNING'
        self.save(update_fields=['status'])

    def mark_success(self, chunk_count: int, total_tokens: int):
        """
        Called when the pipeline completes without errors.
        Stores the number of chunks and total tokens processed.
        """
        self.status = 'SUCCESS'
        self.chunk_count = chunk_count
        self.total_tokens = total_tokens
        self.save(update_fields=['status', 'chunk_count', 'total_tokens'])

    def mark_error(self, message: str):
        """Called if any exception bubbles up during processing."""
        self.status = 'ERROR'
        self.error_message = message
        self.save(update_fields=['status', 'error_message'])
