from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView                    # Base class for custom DRF views
from rest_framework.response import Response                # Used to build API responses
from rest_framework import status                           # HTTP status codes
from .serializers import FileUploadSerializer               # The serializer we just created

# API view to handle file uploads
class FileUploadView(APIView):

    # Handle POST requests to upload a file
    def post(self, request):
        serializer = FileUploadSerializer(data=request.data)  # Bind incoming data to serializer

        # Validate and save if valid
        if serializer.is_valid():
            serializer.save()  # Save to database
            return Response(
                {"message": "File uploaded successfully!", "data": serializer.data},
                status=status.HTTP_201_CREATED
            )

        # If validation fails, return error messages
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
