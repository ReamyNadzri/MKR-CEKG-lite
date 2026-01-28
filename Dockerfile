# Multi-stage build: React frontend + Flask backend
FROM node:18-alpine AS frontend-build

# Build React frontend
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Python backend stage
FROM python:3.9-slim

# Install system dependencies for PIL/Pillow
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Flask app
COPY app.py .
COPY kuih_recognition_model.keras .
COPY templates ./templates
COPY docker-entrypoint.sh /docker-entrypoint.sh

# Make entrypoint script executable
RUN chmod +x /docker-entrypoint.sh

# Copy React build from frontend stage
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Create uploads and feedback directories
RUN mkdir -p uploads feedback_images

# Set default port (Railway will override with $PORT env var)
ENV PORT=8080

# Expose port
EXPOSE 8080

# Use entrypoint script
ENTRYPOINT ["/docker-entrypoint.sh"]
