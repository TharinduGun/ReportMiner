from django.urls import path
from .views import QAAPIView

urlpatterns = [
    path("ask/", QAAPIView.as_view(), name="nl-query"),
]