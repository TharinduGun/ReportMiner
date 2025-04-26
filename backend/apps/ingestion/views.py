from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView                    # Base class for custom DRF views
from rest_framework.response import Response                # Used to build API responses
from rest_framework import status                           # HTTP status codes
from .serializers import FileUploadSerializer               # serializer 
from .extractor import extract_text_from_file
from .utils import preview_text





# API view to handle file uploads
class FileUploadView(APIView):

    # Handle POST requests to upload a file
    def post(self, request):
        serializer = FileUploadSerializer(data=request.data)  # Bind incoming data to serializer

        # Validate and save if valid
        if serializer.is_valid():
            instance = serializer.save()  # Save to database

            file_path = instance.file.path #absolute file path
            file_type = instance.type  # 'pdf', 'docx', or 'xlsx'

             #Extract text based on file type
            text = extract_text_from_file(file_path, file_type)

            return Response({
                "message": "File uploaded and text extracted successfully!",
                "filename": instance.filename,
                "type": instance.type,
                "text_preview": preview_text(text),  # Return a preview of the text
            }, status=status.HTTP_201_CREATED)

            # return Response(
            #     {"message": "File uploaded successfully!", "data": serializer.data},
            #     status=status.HTTP_201_CREATED
            # )

        # If validation fails, return error messages
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
