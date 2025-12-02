# Use Python slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (updated for Debian Trixie)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p static/recordings static/screenshots static/uploads data logs

# Expose port
EXPOSE 10000

# Run the application with gunicorn
CMD gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT main:app
