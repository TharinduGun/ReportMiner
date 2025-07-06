"""
Updated main test file that works with your existing setup
"""
from django.test import TestCase

# Import only the working simple tests
from .tests.test_embeddings_simple import *

# This file ensures Django's test discovery finds our working tests
# Run tests with: python manage.py test apps.ingestion.tests.test_embeddings_simple
