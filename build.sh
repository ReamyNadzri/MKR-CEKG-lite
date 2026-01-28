#!/bin/bash
# Build script for Railway deployment

echo "Building React frontend..."
cd frontend
npm install
npm run build
cd ..

echo "Build complete!"
