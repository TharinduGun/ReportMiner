from rest_framework import serializers

class QuestionSerializer(serializers.Serializer):
    question = serializers.CharField()

class AnswerSerializer(serializers.Serializer):
    answer = serializers.CharField()
    sources = serializers.ListField(
        child=serializers.DictField(),  # e.g. {'chunk_id': ..., 'text': ...}
    )
