"""
Django settings for reportminer project.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Security
SECRET_KEY = 'django-insecure-hd$eis5@(s+)2^65-atcekkfvvy4)pf&7u@8$n*e-(99&jp@nc'
DEBUG = True
ALLOWED_HOSTS = []

# Installed apps
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'corsheaders',

    # Custom apps
    'apps.ingestion',
    'apps.query',

    # Third-party apps
    'rest_framework',
    'django_extensions',
]

# Middleware
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'corsheaders.middleware.CorsMiddleware',
]

CORS_ALLOW_ALL_ORIGINS = True

# URL and WSGI
ROOT_URLCONF = 'reportminer.urls'
WSGI_APPLICATION = 'reportminer.wsgi.application'

# Templates
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

# Database (SQLite for development)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files
STATIC_URL = 'static/'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Django REST Framework configuration
REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny',
    ],
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
}

# OpenAI + Chat Model settings
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
CHAT_MODEL_NAME = os.getenv('CHAT_MODEL_NAME', 'gpt-4o')

# ChromaDB settings
CHROMA_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR",
    str(BASE_DIR / "chroma_data")
)
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "reportminer")

# Celery (Redis as broker)
CELERY_BROKER_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = CELERY_BROKER_URL

# Poppler path for PDF rendering (required for PyMuPDF or similar PDF tools)
poppler_path = r"C:\Program Files\poppler-24.08.0\Library\bin"
if poppler_path not in os.environ.get('PATH', ''):
    os.environ['PATH'] = poppler_path + os.pathsep + os.environ['PATH']

# Chunking configuration (used by splitter.py)
INGESTION_ROW_EMBED_THRESHOLD = 200
INGESTION_ROW_GROUP_SIZE = 50

# CSV/Excel ingestion behavior
CSV_FULL_SHEET_INGESTION = False
CSV_CHUNKSIZE = 50_000
EXCEL_FULL_SHEET_INGESTION = True
