from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import DocumentUploadSerializer
from .tasks import process_document

class DocumentUploadAPIView(APIView):
    """
    POST /api/ingestion/upload/
    Accepts a file upload, creates a Document record (status=PENDING),
    enqueues the processing task, and returns the document ID.

    Note: permission handling is omitted for now.
    """
    def post(self, request, format=None):
        serializer = DocumentUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        # Create the Document (status=PENDING)
        document = serializer.save()

        # Enqueue Celery task
        process_document.delay(str(document.id))

        # Return the document ID for status polling
        return Response(
            {"id": document.id, "status": document.status},
            status=status.HTTP_202_ACCEPTED
        )
