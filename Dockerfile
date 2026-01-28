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
COPY templates/ ./templates/
COPY static/ ./static/

# Copy React build from frontend stage
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Create uploads and feedback directories
RUN mkdir -p uploads feedback

# Expose port (Railway will use $PORT)
EXPOSE 8080

# Start gunicorn
CMD gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 app:app
