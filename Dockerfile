FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY feature_extractor.py .
COPY heart_rate_detector.py .
COPY stress_feature_extractor.py .
COPY stress_heart_rate_detector.py .

# Create models directory
RUN mkdir -p models

# Copy model file if it exists (will be added later)
# COPY models/catboost_bp_model.pkl models/catboost_bp_model.pkl

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Set environment variables
ENV PORT=7860
ENV BP_MODEL_PATH=/app/models/bp_gradientboost.pkl
ENV BP_SCALER_PATH=/app/models/bp_scaler.pkl
ENV STRESS_MODEL_PATH=/app/models/stress_gradientboost.pkl
ENV STRESS_SCALER_PATH=/app/models/stress_scaler.pkl
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/health')"

# Run the application
CMD ["python", "app.py"]
