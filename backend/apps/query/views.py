from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .services import run_query

class QueryAPIView(APIView):
    """
    POST { "question": "..." } → { "answer": "...", "sources": [...] }
    """
    def post(self, request):
        question = request.data.get("question")
        if not question:
            return Response(
                {"detail": "Missing 'question' in request body."},
                status=status.HTTP_400_BAD_REQUEST
            )
        output = run_query(question)
        return Response(output)