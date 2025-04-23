from rest_framework import serializers  # DRF serializer base classes
from .models import FileUpload          # Import the FileUpload model

# This serializer converts the FileUpload model into JSON format
# It also validates incoming data before saving to the database
class FileUploadSerializer(serializers.ModelSerializer):
    
    # Meta class defines which model and which fields to include
    class Meta:
        model = FileUpload  # This is the Django model to serialize
        fields = ['id', 'filename', 'file', 'type', 'uploaded_at']  # Include all model fields: filename, file, type, uploaded_at
