from rest_framework import serializers
from .models import Document

class DocumentUploadSerializer(serializers.ModelSerializer):
    """
    Serializer for uploading documents to the ingestion pipeline.

    - Validates file type via model validators
    - Returns document ID upon successful creation
    """
    class Meta:
        model = Document
        fields = ['id', 'file']
        read_only_fields = ['id']

    def create(self, validated_data):
        # Save the Document instance with default status = PENDING
        document = Document.objects.create(**validated_data)
        return document
