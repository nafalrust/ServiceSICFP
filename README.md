# Combined AI Service - Blood Pressure & Stress Detection

Unified AI service providing both blood pressure classification and stress detection using Gradient Boosting models trained on PPG and physiological signals.

## Features

- 🩺 **6-Class Blood Pressure Classification**: Hypotension, Normal, Elevated, Hypertension Stage 1, Hypertension Stage 2, Hypertensive Crisis
- 🧘 **2-Class Stress Detection**: Baseline (No Stress) vs Stress
- 💓 **Heart Rate & HRV Analysis**: Automatic HR and HRV metrics extraction
- 📊 **Multi-Modal Feature Extraction**: 17 features (BP) + 21 features (Stress)
- 🚀 **Fast Inference**: Gradient Boosting models optimized for production
- 🐳 **Docker Ready**: Easy deployment with Docker
- ☁️ **Hugging Face Compatible**: Ready to deploy on Hugging Face Spaces
- ✨ **Flexible Inputs**: PPG only for BP, PPG + Temperature for stress

## Quick Start

### Local Development

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Place your trained models**:
```bash
mkdir -p models
# Copy your models to models/
# - gradient_boosting_bp_model.pkl
# - bp_scaler.pkl
# - gradient_boosting_stress_model.pkl
# - stress_scaler.pkl
```

3. **Run the service**:
```bash
python app.py
```

The service will start on `http://localhost:7860`

### Docker Deployment

1. **Build the Docker image**:
```bash
docker build -t bp-classifier .
```

2. **Run the container**:
```bash
docker run -p 7860:7860 -v $(pwd)/models:/app/models bp-classifier
```

### Hugging Face Spaces Deployment

1. **Create a new Space** on Hugging Face with Docker SDK

2. **Add these files to your Space**:
   - `app.py`
   - `feature_extractor.py`
   - `heart_rate_detector.py`
   - `requirements.txt`
   - `Dockerfile`
   - `models/catboost_bp_model.pkl`

3. **The Space will automatically deploy**

## API Endpoints

### 1. Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "bp_model_loaded": true,
  "bp_scaler_loaded": true,
  "stress_model_loaded": true,
  "stress_scaler_loaded": true,
  "service": "Combined Blood Pressure & Stress Detection"
}
```

### 2. Predict Blood Pressure
```bash
POST /predict_bp
```

Request body:
```json
{
  "ppg": [array of 1000 PPG values @ 100Hz]
}
```

Response:
```json
{
  "prediction": "Normal",
  "class_id": 1,
  "confidence": 0.92,
  "probabilities": [0.02, 0.92, 0.03, 0.02, 0.01, 0.00],
  "heart_rate": {
    "mean_bpm": 72.5,
    "std_bpm": 3.2,
    "min_bpm": 68.0,
    "max_bpm": 78.0,
    "num_peaks": 12
  },
  "features_extracted": 17
}
```

### 3. Predict Stress Level
```bash
POST /predict_stress
```

Request body:
```json
{
  "ppg": [array of 640 PPG values @ 64Hz],
  "temperature": [array of 40 temperature values @ 4Hz]
}
```

Response:
```json
{
  "prediction": "Stress",
  "class_id": 1,
  "confidence": 0.88,
  "probabilities": [0.12, 0.88],
  "heart_rate": {
    "mean_bpm": 85.2,
    "std_bpm": 5.3,
    "min_bpm": 78.0,
    "max_bpm": 95.0,
    "rmssd_ms": 25.4,
    "sdnn_ms": 42.1
  },
  "features_extracted": 21
}
```

### 4. Extract Features Only (BP)
```bash
POST /extract_features
```

Request body: Same as `/predict`

Response:
```json
{
  "features": {
    "ppg_mean": 0.5,
    "ppg_std": 0.2,
    ...
  },
  "heart_rate": {
    "hr_mean": 72.5,
    ...
  }
}
```

### 4. Service Info
```bash
GET /info
```

Response: Service metadata and usage information

## Input Requirements

### PPG Signal
- **Length**: Minimum 100 samples
- **Recommended**: 1000 samples (10 seconds at 100 Hz - matches training data)
- **Format**: Array of numeric values
- **Sampling Rate**: 100 Hz (from training dataset)

No patient demographics required! Features are extracted from PPG signal and heart rate only.

## Classification Classes

### Blood Pressure (6 Classes)

| Class ID | Label | Description |
|----------|-------|-------------|
| 0 | Hypotension | SBP < 90 or DBP < 60 |
| 1 | Normal | SBP < 120 and DBP < 80 |
| 2 | Elevated | SBP 120-129 and DBP < 80 |
| 3 | Hypertension Stage 1 | SBP 130-139 or DBP 80-89 |
| 4 | Hypertension Stage 2 | SBP ≥ 140 or DBP ≥ 90 |
| 5 | Hypertensive Crisis | SBP > 180 or DBP > 120 |

### Stress Detection (2 Classes)

| Class ID | Label | Description |
|----------|-------|-------------|
| 0 | Baseline | No stress condition (relaxed state) |
| 1 | Stress | Stressed condition (elevated physiological response) |

## Feature Extraction Details

### Blood Pressure Features (17 Total)

#### Signal Preprocessing

Before feature extraction, all PPG signals are preprocessed with:

- **Bandpass Filter**: 0.5-8.0 Hz
  - Removes baseline drift (< 0.5 Hz)
  - Removes high-frequency noise (> 8.0 Hz)
  - 4th order Butterworth filter
  - Zero-phase filtering (forward-backward)

This matches the preprocessing pipeline used in the training notebook.

### Total: 17 Features

1. **Time Domain (9 features)**:
   - Mean, Std, Min, Max, Median, Range, RMS, Skewness, Kurtosis

2. **Peak-Based (6 features)**:
   - Number of peaks
   - Mean peak amplitude, Std peak amplitude
   - Mean peak interval, Std peak interval
   - Heart rate estimate (BPM)

3. **Derivative (2 features)**:
   - Mean first derivative
   - Mean second derivative

All features are automatically extracted from the filtered PPG signal. No manual feature engineering required!

### Stress Detection Features (21 Total)

1. **PPG Statistical (6 features)**:
   - Mean, Std, Min, Max, Median, RMS

2. **PPG Frequency (3 features)**:
   - Dominant frequency, Frequency energy, Frequency mean

3. **Temperature Statistical (6 features)**:
   - Mean, Std, Min, Max, Median, RMS

4. **Heart Rate & HRV (6 features)**:
   - HR Mean, HR Std, HR Min, HR Max, RMSSD, SDNN

All features extracted from PPG (64 Hz) and Temperature (4 Hz) signals.

## Example Usage

### Python - Blood Pressure
```python
import requests
import numpy as np

# Generate sample PPG signal (1000 samples, 10 seconds at 100 Hz)
ppg_signal = np.sin(np.linspace(0, 10*np.pi, 1000)) + np.random.normal(0, 0.1, 1000)

data = {
    "ppg": ppg_signal.tolist()
}

response = requests.post("http://localhost:7860/predict_bp", json=data)
print(response.json())
```

### Python - Stress Detection
```python
import requests
import numpy as np

# Generate sample signals
ppg_signal = np.sin(np.linspace(0, 10*np.pi, 640)) + np.random.normal(0, 0.1, 640)
temp_signal = 36.5 + np.random.normal(0, 0.2, 40)

data = {
    "ppg": ppg_signal.tolist(),
    "temperature": temp_signal.tolist()
}

response = requests.post("http://localhost:7860/predict_stress", json=data)
print(response.json())
```

### cURL
```bash
curl -X POST http://localhost:7860/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ppg": [0.1, 0.2, ..., 0.5]
  }'
```

### JavaScript
```javascript
const data = {
  ppg: Array(1000).fill(0).map(() => Math.random())
};

fetch('http://localhost:7860/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify(data)
})
.then(response => response.json())
.then(data => console.log(data));
```

## Model Information

### Blood Pressure Classification
- **Algorithm**: Gradient Boosting Classifier
- **Features**: 17 extracted features (time-domain, peak-based, derivatives)
- **Classes**: 6 (Hypotension, Normal, Elevated, Hypertension Stage 1, Hypertension Stage 2, Hypertensive Crisis)
- **Input**: PPG signal only (no demographics needed)
- **Sampling Rate**: 100 Hz
- **Window Size**: 1000 samples (10 seconds)
- **Feature Scaling**: StandardScaler

### Stress Detection
- **Algorithm**: Gradient Boosting Classifier
- **Dataset**: WESAD (Wearable Stress and Affect Detection)
- **Features**: 21 features (PPG + Temperature + HRV)
- **Classes**: 2 (Baseline, Stress)
- **Input**: PPG + Temperature signals
- **Sampling Rates**: PPG 64 Hz, Temperature 4 Hz
- **Window Size**: PPG 640 samples, Temperature 40 samples (10 seconds)
- **Feature Scaling**: StandardScaler

## Development

### Project Structure
```
ServiceAI/
├── app.py                         # Flask API server (combined)
├── feature_extractor.py           # BP feature extraction
├── heart_rate_detector.py         # BP heart rate detection
├── stress_feature_extractor.py    # Stress feature extraction
├── stress_heart_rate_detector.py  # Stress heart rate detection
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker configuration
├── README.md                      # This file
└── models/
    ├── gradient_boosting_bp_model.pkl     # BP model
    ├── bp_scaler.pkl                      # BP scaler
    ├── gradient_boosting_stress_model.pkl # Stress model
    └── stress_scaler.pkl                  # Stress scaler
```

### Adding Your Models

After training your models in the Jupyter notebooks:

```python
# In your BP training notebook
import joblib

# Save the trained BP model and scaler
joblib.dump(gb_bp_model, 'ServiceAI/models/gradient_boosting_bp_model.pkl')
joblib.dump(bp_scaler, 'ServiceAI/models/bp_scaler.pkl')

# In your Stress training notebook
# Save the trained Stress model and scaler
joblib.dump(gb_stress_model, 'ServiceAI/models/gradient_boosting_stress_model.pkl')
joblib.dump(stress_scaler, 'ServiceAI/models/stress_scaler.pkl')
```

## Environment Variables

- `PORT`: Server port (default: 7860)
- `BP_MODEL_PATH`: Path to BP model file (default: ./models/gradient_boosting_bp_model.pkl)
- `BP_SCALER_PATH`: Path to BP scaler file (default: ./models/bp_scaler.pkl)
- `STRESS_MODEL_PATH`: Path to Stress model file (default: ./models/gradient_boosting_stress_model.pkl)
- `STRESS_SCALER_PATH`: Path to Stress scaler file (default: ./models/stress_scaler.pkl)

## Troubleshooting

### Model not loading
- Ensure `gradient_boosting_bp_model.pkl` and `scaler.pkl` exist in `models/` directory
- Check file permissions
- Verify models were saved with `joblib.dump()`

### PPG signal too short error
- Provide at least 64 samples (minimum 1 second)
- Recommended: 640 samples (10 seconds)

### Feature extraction errors
- Ensure PPG signal contains numeric values
- Check for NaN or infinite values in input
- Validate signal is 1D array

## License

MIT License - See LICENSE file for details

## Citation

If you use this service in your research, please cite:

```bibtex
@article{bp_classification_2024,
  title={Blood Pressure Classification from PPG Signals using Gradient Boosting},
  author={Your Name},
  year={2024}
}
```

## Support

For issues and questions:
- Open an issue on GitHub
- Contact: your-email@example.com

## Acknowledgments

- Based on research in cuff-less blood pressure estimation
- Uses CatBoost for efficient gradient boosting
- PPG signal processing techniques from biomedical literature
