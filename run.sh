#!/bin/bash

# Build and run script for Combined AI Service (BP + Stress Detection)
# Supports multiple modes: docker, local, dev, stop, rebuild

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Service configuration
SERVICE_NAME="combined-ai-service"
DOCKER_IMAGE="combined-ai:latest"
PORT=7860

# Print colored output
print_header() {
    echo -e "${BLUE}========================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if models exist
check_models() {
    local missing_models=0
    
    echo ""
    print_info "Checking model files..."
    
    # BP models
    if [ ! -f "models/bp_gradientboost.pkl" ]; then
        print_warning "BP model not found: models/bp_gradientboost.pkl"
        missing_models=$((missing_models + 1))
    else
        print_success "BP model found"
    fi
    
    if [ ! -f "models/bp_scaler.pkl" ]; then
        print_warning "BP scaler not found: models/bp_scaler.pkl"
        missing_models=$((missing_models + 1))
    else
        print_success "BP scaler found"
    fi
    
    # Stress models
    if [ ! -f "models/stress_gradientboost.pkl" ]; then
        print_warning "Stress model not found: models/stress_gradientboost.pkl"
        missing_models=$((missing_models + 1))
    else
        print_success "Stress model found"
    fi
    
    if [ ! -f "models/stress_scaler.pkl" ]; then
        print_warning "Stress scaler not found: models/stress_scaler.pkl"
        missing_models=$((missing_models + 1))
    else
        print_success "Stress scaler found"
    fi
    
    if [ $missing_models -gt 0 ]; then
        echo ""
        print_warning "$missing_models model file(s) missing"
        print_info "Train and export models from Jupyter notebooks first"
        echo ""
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Build Docker image
build_docker() {
    print_header "Building Docker Image"
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    echo ""
    print_info "Building image: $DOCKER_IMAGE"
    docker build -t $DOCKER_IMAGE .
    
    if [ $? -ne 0 ]; then
        print_error "Docker build failed"
        exit 1
    fi
    
    print_success "Docker image built successfully"
}

# Run with Docker
run_docker() {
    print_header "Starting Service with Docker"
    
    check_models
    
    # Stop existing container if running
    if docker ps -a --format '{{.Names}}' | grep -q "^${SERVICE_NAME}$"; then
        print_info "Stopping existing container..."
        docker stop $SERVICE_NAME >/dev/null 2>&1 || true
        docker rm $SERVICE_NAME >/dev/null 2>&1 || true
    fi
    
    # Check if image exists, build if not
    if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^${DOCKER_IMAGE}$"; then
        print_info "Docker image not found, building..."
        build_docker
    fi
    
    echo ""
    print_info "Starting container on port $PORT..."
    docker run -d \
        --name $SERVICE_NAME \
        -p $PORT:$PORT \
        -v "$(pwd)/models:/app/models" \
        --restart unless-stopped \
        $DOCKER_IMAGE
    
    if [ $? -ne 0 ]; then
        print_error "Failed to start service"
        exit 1
    fi
    
    echo ""
    print_success "Service started successfully!"
    print_service_info
}

# Run locally (without Docker)
run_local() {
    print_header "Starting Service Locally"
    
    check_models
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    print_info "Activating virtual environment..."
    source venv/bin/activate
    
    # Install dependencies
    print_info "Installing dependencies..."
    pip install -q --upgrade pip
    pip install -q -r requirements.txt
    
    echo ""
    print_success "Starting Flask server on port $PORT..."
    echo ""
    export PORT=$PORT
    python app.py
}

# Run in development mode with auto-reload
run_dev() {
    print_header "Starting Service in Development Mode"
    
    check_models
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    print_info "Activating virtual environment..."
    source venv/bin/activate
    
    # Install dependencies
    print_info "Installing dependencies..."
    pip install -q --upgrade pip
    pip install -q -r requirements.txt
    
    echo ""
    print_success "Starting Flask development server with auto-reload..."
    echo ""
    export FLASK_APP=app.py
    export FLASK_ENV=development
    export PORT=$PORT
    flask run --host=0.0.0.0 --port=$PORT --reload
}

# Stop service
stop_service() {
    print_header "Stopping Service"
    
    if docker ps --format '{{.Names}}' | grep -q "^${SERVICE_NAME}$"; then
        print_info "Stopping Docker container..."
        docker stop $SERVICE_NAME
        docker rm $SERVICE_NAME
        print_success "Service stopped"
    else
        print_warning "Service is not running"
    fi
}

# Test service
test_service() {
    print_header "Testing Service"
    
    # Check if service is running
    if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        print_error "Service is not running on port $PORT"
        exit 1
    fi
    
    print_info "Running test suite..."
    
    if [ -f "test_combined_service.py" ]; then
        python3 test_combined_service.py
    else
        print_warning "Test file not found: test_combined_service.py"
    fi
}

# Print service information
print_service_info() {
    echo ""
    echo -e "${GREEN}========================================================================${NC}"
    echo -e "${GREEN}Service Information${NC}"
    echo -e "${GREEN}========================================================================${NC}"
    echo ""
    echo -e "${BLUE}Service URL:${NC} http://localhost:$PORT"
    echo ""
    echo -e "${BLUE}Endpoints:${NC}"
    echo "  Health Check:    GET  http://localhost:$PORT/health"
    echo "  Service Info:    GET  http://localhost:$PORT/info"
    echo "  Predict BP:      POST http://localhost:$PORT/predict_bp"
    echo "  Predict Stress:  POST http://localhost:$PORT/predict_stress"
    echo "  Extract Features:POST http://localhost:$PORT/extract_features"
    echo ""
    echo -e "${BLUE}Useful Commands:${NC}"
    echo "  View logs:       docker logs -f $SERVICE_NAME"
    echo "  Stop service:    docker stop $SERVICE_NAME"
    echo "  Remove:          docker rm $SERVICE_NAME"
    echo "  Test service:    python3 test_combined_service.py"
    echo "  Rebuild:         ./run.sh rebuild"
    echo ""
    echo -e "${GREEN}========================================================================${NC}"
}

# Show usage
show_usage() {
    echo ""
    echo "Usage: ./run.sh [MODE]"
    echo ""
    echo "Modes:"
    echo "  docker    - Run with Docker (default, production-ready)"
    echo "  local     - Run locally with Python virtual environment"
    echo "  dev       - Run in development mode with auto-reload"
    echo "  build     - Build Docker image only"
    echo "  rebuild   - Rebuild Docker image and run"
    echo "  stop      - Stop running service"
    echo "  test      - Test the running service"
    echo "  help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh docker    # Run with Docker"
    echo "  ./run.sh local     # Run locally"
    echo "  ./run.sh dev       # Run in dev mode"
    echo "  ./run.sh rebuild   # Rebuild and run"
    echo ""
}

# Main script
main() {
    MODE="${1:-docker}"  # Default to docker mode
    
    case $MODE in
        docker)
            run_docker
            ;;
        local)
            run_local
            ;;
        dev)
            run_dev
            ;;
        build)
            build_docker
            ;;
        rebuild)
            build_docker
            run_docker
            ;;
        stop)
            stop_service
            ;;
        test)
            test_service
            ;;
        help|-h|--help)
            show_usage
            ;;
        *)
            print_error "Unknown mode: $MODE"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
