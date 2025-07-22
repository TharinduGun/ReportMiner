FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    pkg-config \
    libpoppler-cpp-dev \
    libpoppler-glib-dev \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Create app directory
WORKDIR /app

# Copy and install Python dependencies first (for better caching)
COPY backend/requirements-minimal.txt ./backend/
RUN pip install --no-cache-dir -r ./backend/requirements-minimal.txt

# Copy and build frontend
COPY report-miner-frontend/ ./frontend/
WORKDIR /app/frontend
RUN npm install && npm run build

# Copy backend code
WORKDIR /app
COPY backend/ ./backend/

# Create directories for uploads and data
RUN mkdir -p ./backend/documents ./backend/chroma_db ./chroma_data

# Set environment variables
ENV PYTHONPATH="/app/backend:/app/backend/apps"
ENV DJANGO_SETTINGS_MODULE="reportminer.settings"

# Expose port
EXPOSE 8000

# Change to backend directory
WORKDIR /app/backend

# Run Django migrations and start server
CMD ["sh", "-c", "python manage.py migrate && python manage.py runserver 0.0.0.0:8000"]