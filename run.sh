#!/bin/bash

# Build and run script for local development

echo "========================================================================"
echo "Blood Pressure Classification Service - Build & Run"
echo "========================================================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null
then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if model file exists
if [ ! -f "models/catboost_bp_model.pkl" ]; then
    echo "⚠️  Warning: Model file not found at models/catboost_bp_model.pkl"
    echo "Please train and export your model first."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]
    then
        exit 1
    fi
fi

# Build Docker image
echo ""
echo "Building Docker image..."
docker build -t bp-classifier:latest .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed"
    exit 1
fi

echo ""
echo "✓ Docker image built successfully"
echo ""

# Run Docker container
echo "Starting service on port 7860..."
docker run -d \
    --name bp-classifier-service \
    -p 7860:7860 \
    -v $(pwd)/models:/app/models \
    bp-classifier:latest

if [ $? -ne 0 ]; then
    echo "❌ Failed to start service"
    exit 1
fi

echo ""
echo "========================================================================"
echo "✓ Service started successfully!"
echo "========================================================================"
echo ""
echo "Service URL: http://localhost:7860"
echo ""
echo "Useful commands:"
echo "  - View logs:     docker logs -f bp-classifier-service"
echo "  - Stop service:  docker stop bp-classifier-service"
echo "  - Remove:        docker rm bp-classifier-service"
echo "  - Test service:  python test_service.py"
echo ""
echo "Endpoints:"
echo "  - Health:        GET  http://localhost:7860/health"
echo "  - Predict:       POST http://localhost:7860/predict"
echo "  - Features:      POST http://localhost:7860/extract_features"
echo "  - Info:          GET  http://localhost:7860/info"
echo ""
echo "========================================================================"
