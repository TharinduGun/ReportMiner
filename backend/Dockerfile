# Use official Python image
FROM python:3.11

# Set working directory inside the container
WORKDIR /app

# Copy requirements.txt from local backend dir to container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all backend project files
COPY . .

# Expose Django port
EXPOSE 8000

# Default command to run the server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
