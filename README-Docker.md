# ReportMiner Docker Setup

This guide explains how to run ReportMiner using Docker for easy deployment and development across different laptops.

## Prerequisites

- Docker Desktop installed on your system
- Docker Compose (usually included with Docker Desktop)
- At least 4GB RAM available for containers

## Quick Start

### Option 1: Using the startup scripts

**Windows:**
```cmd
.\start-docker.bat
```

**Linux/macOS:**
```bash
./start-docker.sh
```

### Option 2: Manual Docker commands

1. **Build the containers:**
   ```bash
   docker-compose build
   ```

2. **Start the services:**
   ```bash
   docker-compose up -d
   ```

3. **View logs:**
   ```bash
   docker-compose logs -f
   ```

## Services

The Docker setup includes:

- **web**: Django backend API server (port 8000)
- **celery**: Background task processor
- **celery-beat**: Scheduled task runner
- **redis**: Message broker and cache

## Accessing the Application

- **Backend API**: http://localhost:8000
- **Admin Interface**: http://localhost:8000/admin
- **Frontend**: Served through the Django static files

## Development vs Production

### Development Mode (docker-compose.dev.yml)
```bash
docker-compose -f docker-compose.dev.yml up -d
```
- Hot reloading enabled
- Local files mounted as volumes
- Debug mode enabled

### Production Mode (docker-compose.yml)
```bash
docker-compose up -d
```
- Optimized for performance
- No file mounting
- Production settings

## Data Persistence

The following data is persisted in Docker volumes:
- SQLite database (`backend/db.sqlite3`)
- Uploaded documents (`backend/documents/`)
- ChromaDB data (`chroma_data/`)
- Redis data (in named volume)

## Common Commands

```bash
# Start services in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and start
docker-compose up --build

# Run Django commands inside container
docker-compose exec web python manage.py createsuperuser
docker-compose exec web python manage.py migrate
docker-compose exec web python manage.py collectstatic

# Access container shell
docker-compose exec web bash

# View running containers
docker-compose ps
```

## Troubleshooting

### Build Issues
If the build fails due to package installation:
```bash
docker-compose build --no-cache
```

### Permission Issues (Linux/macOS)
```bash
sudo chown -R $USER:$USER ./backend/documents ./chroma_data
```

### Port Conflicts
If port 8000 is already in use, modify the port mapping in `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Use port 8001 instead
```

### Memory Issues
If containers are running out of memory, increase Docker's memory limit in Docker Desktop settings.

## Environment Variables

You can customize the setup using environment variables:

```bash
# .env file
DEBUG=1
DJANGO_SECRET_KEY=your-secret-key-here
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0
```

## Cleanup

To remove all containers, volumes, and images:
```bash
docker-compose down -v
docker system prune -a
```

## Moving Between Laptops

1. **Export your data:**
   ```bash
   # Copy the entire project directory including:
   # - backend/db.sqlite3 (database)
   # - backend/documents/ (uploaded files)
   # - chroma_data/ (vector database)
   ```

2. **On the new laptop:**
   ```bash
   # Copy the project directory
   # Install Docker Desktop
   # Run the application
   docker-compose up -d
   ```

Your application state will be preserved across different machines!