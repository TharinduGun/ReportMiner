from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from .serializers import QuestionSerializer, AnswerSerializer
from .services.rag_chain import build_qa_chain


class QAAPIView(APIView):
    permission_classes = [AllowAny]


    def post(self, request):
        serializer = QuestionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        question = serializer.validated_data["question"]


        qa_chain = build_qa_chain()
        result = qa_chain({"query": question})


        sources = [
            {"chunk_id": doc.metadata.get("chunk_id"), "text": doc.page_content}
            for doc in result.get("source_documents", [])
        ]
        answer = {"answer": result.get("result"), "sources": sources}
        return Response(AnswerSerializer(answer).data, status=status.HTTP_200_OK)