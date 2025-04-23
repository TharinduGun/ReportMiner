from django.urls import path
from .views import FileUploadView

# route for ingestion app
urlpatterns = [
    path('upload/', FileUploadView.as_view(), name='file-upload'),
]
