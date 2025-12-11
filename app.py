"""
Combined AI Service for Blood Pressure and Stress Classification
Using Gradient Boosting models trained on PPG signals
"""

from flask import Flask, request, jsonify
import numpy as np
import joblib
import os
from feature_extractor import PPGFeatureExtractor
from heart_rate_detector import HeartRateDetector
from stress_feature_extractor import StressFeatureExtractor
from stress_heart_rate_detector import HeartRateDetector as StressHeartRateDetector

app = Flask(__name__)

# Load Blood Pressure models
BP_MODEL_PATH = os.environ.get('BP_MODEL_PATH', './models/bp_gradientboost.pkl')
BP_SCALER_PATH = os.environ.get('BP_SCALER_PATH', './models/bp_scaler.pkl')
bp_model = None
bp_scaler = None
bp_feature_extractor = PPGFeatureExtractor(fs=100)
bp_hr_detector = HeartRateDetector(fs=100)

# Load Stress Detection models
STRESS_MODEL_PATH = os.environ.get('STRESS_MODEL_PATH', './models/stress_gradientboost.pkl')
STRESS_SCALER_PATH = os.environ.get('STRESS_SCALER_PATH', './models/stress_scaler.pkl')
stress_model = None
stress_scaler = None
stress_feature_extractor = StressFeatureExtractor(fs=64)
stress_hr_detector = StressHeartRateDetector(fs=64)

def load_models():
    """Load all trained models and scalers."""
    global bp_model, bp_scaler, stress_model, stress_scaler
    
    # Load BP model (with feature extractor dependency)
    try:
        # Import feature extractor to satisfy pickle dependencies
        import sys
        from feature_extractor import PPGFeatureExtractor as _PPGFeatureExtractor
        sys.modules['__main__'].PPGFeatureExtractor = _PPGFeatureExtractor
        
        bp_model = joblib.load(BP_MODEL_PATH)
        
        # If model is dict (contains 'model' key), extract the actual model
        if isinstance(bp_model, dict) and 'model' in bp_model:
            bp_model = bp_model['model']
        
        print(f"✓ BP Model loaded successfully from {BP_MODEL_PATH}")
    except Exception as e:
        print(f"⚠ Warning: Could not load BP model - {str(e)}")
        bp_model = None
    
    try:
        bp_scaler = joblib.load(BP_SCALER_PATH)
        print(f"✓ BP Scaler loaded successfully from {BP_SCALER_PATH}")
    except Exception as e:
        print(f"⚠ Warning: Could not load BP scaler - {str(e)}")
        bp_scaler = None
    
    # Load Stress model
    try:
        stress_model = joblib.load(STRESS_MODEL_PATH)
        print(f"✓ Stress Model loaded successfully from {STRESS_MODEL_PATH}")
    except Exception as e:
        print(f"⚠ Warning: Could not load Stress model - {str(e)}")
        stress_model = None
    
    try:
        stress_scaler = joblib.load(STRESS_SCALER_PATH)
        print(f"✓ Stress Scaler loaded successfully from {STRESS_SCALER_PATH}")
    except Exception as e:
        print(f"⚠ Warning: Could not load Stress scaler - {str(e)}")
        stress_scaler = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'bp_model_loaded': bp_model is not None,
        'bp_scaler_loaded': bp_scaler is not None,
        'stress_model_loaded': stress_model is not None,
        'stress_scaler_loaded': stress_scaler is not None,
        'service': 'Combined Blood Pressure & Stress Detection'
    })

@app.route('/predict_bp', methods=['POST'])
def predict_bp():
    """
    Predict blood pressure class from PPG signal only (no demographics needed).
    
    Expected JSON input:
    {
        "ppg": [array of PPG values @ 100Hz]
    }
    
    Returns:
    {
        "prediction": "Normal",
        "class_id": 2,
        "confidence": 0.85,
        "features": {...}
    }
    """
    if bp_model is None:
        return jsonify({'error': 'BP model not loaded'}), 500
    
    if bp_scaler is None:
        return jsonify({'error': 'BP scaler not loaded'}), 500
    
    try:
        # Parse input
        data = request.get_json()
        
        if 'ppg' not in data:
            return jsonify({'error': 'PPG signal is required'}), 400
        
        ppg_signal = np.array(data['ppg'])
        
        # Validate PPG signal
        if len(ppg_signal) < 64:
            return jsonify({'error': 'PPG signal too short (minimum 64 samples)'}), 400
        
        # Apply bandpass filter to remove noise
        ppg_filtered = bp_feature_extractor.bandpass_filter(ppg_signal)
        
        # Extract heart rate metrics
        hr_metrics = bp_hr_detector.detect_heart_rate(ppg_filtered)
        
        # Extract features (no demographics needed)
        features = bp_feature_extractor.extract_all_features(
            ppg=ppg_filtered,
            hr_metrics=hr_metrics
        )
        
        # Prepare feature vector for prediction
        feature_vector = bp_feature_extractor.features_to_vector(features)
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        # Scale features using the trained scaler
        feature_vector_scaled = bp_scaler.transform(feature_vector)
        
        # Make prediction
        prediction_class = bp_model.predict(feature_vector_scaled)[0]
        
        # Get prediction probabilities if available
        try:
            prediction_proba = bp_model.predict_proba(feature_vector_scaled)[0]
            confidence = float(np.max(prediction_proba))
            probabilities = prediction_proba.tolist()
        except:
            confidence = None
            probabilities = None
        
        # Map class to label (matches training notebook)
        class_labels = {
            0: 'Hypotension',
            1: 'Normal', 
            2: 'Elevated',
            3: 'Hypertension Stage 1',
            4: 'Hypertension Stage 2',
            5: 'Hypertensive Crisis'
        }
        
        prediction_label = class_labels.get(prediction_class, 'Unknown')
        
        # Prepare response
        response = {
            'prediction': prediction_label,
            'class_id': int(prediction_class),
            'confidence': confidence,
            'probabilities': probabilities,
            'heart_rate': {
                'mean_bpm': float(hr_metrics.get('hr_mean', 0)),
                'std_bpm': float(hr_metrics.get('hr_std', 0)),
                'min_bpm': float(hr_metrics.get('hr_min', 0)),
                'max_bpm': float(hr_metrics.get('hr_max', 0)),
                'num_peaks': int(hr_metrics.get('num_peaks', 0))
            },
            'features_extracted': len(feature_vector[0])
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_stress', methods=['POST'])
def predict_stress():
    """
    Predict stress level from PPG and Temperature signals.
    
    Expected JSON input:
    {
        "ppg": [array of 640 PPG values @ 64Hz],
        "temperature": [array of 40 temperature values @ 4Hz]
    }
    
    Returns:
    {
        "prediction": "Baseline" or "Stress",
        "class_id": 0 or 1,
        "confidence": 0.0-1.0,
        "probabilities": [prob_baseline, prob_stress],
        "heart_rate": {...},
        "features_extracted": 21
    }
    """
    if stress_model is None:
        return jsonify({'error': 'Stress model not loaded'}), 500
    
    if stress_scaler is None:
        return jsonify({'error': 'Stress scaler not loaded'}), 500
    
    try:
        # Parse input
        data = request.get_json()
        
        if 'ppg' not in data or 'temperature' not in data:
            return jsonify({'error': 'Both PPG and temperature signals are required'}), 400
        
        ppg_signal = np.array(data['ppg'])
        temp_signal = np.array(data['temperature'])
        
        # Validate signal lengths
        if len(ppg_signal) < 64:
            return jsonify({'error': 'PPG signal too short (minimum 64 samples)'}), 400
        
        if len(temp_signal) < 4:
            return jsonify({'error': 'Temperature signal too short (minimum 4 samples)'}), 400
        
        # Apply bandpass filter to PPG signal to remove noise
        ppg_filtered = stress_feature_extractor.bandpass_filter(ppg_signal)
        
        # Extract heart rate metrics
        hr_metrics = stress_hr_detector.detect_heart_rate(ppg_filtered)
        
        # Extract features
        features = stress_feature_extractor.extract_all_features(
            ppg=ppg_filtered,
            temp=temp_signal,
            hr_metrics=hr_metrics
        )
        
        # Prepare feature vector for prediction
        feature_vector = stress_feature_extractor.features_to_vector(features)
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        # Scale features
        feature_vector_scaled = stress_scaler.transform(feature_vector)
        
        # Make prediction
        prediction_class = stress_model.predict(feature_vector_scaled)[0]
        
        # Get prediction probabilities if available
        try:
            prediction_proba = stress_model.predict_proba(feature_vector_scaled)[0]
            confidence = float(np.max(prediction_proba))
            probabilities = prediction_proba.tolist()
        except:
            confidence = None
            probabilities = None
        
        # Map class to label
        class_labels = {
            0: 'Baseline',
            1: 'Stress'
        }
        
        prediction_label = class_labels.get(prediction_class, 'Unknown')
        
        # Prepare response
        response = {
            'prediction': prediction_label,
            'class_id': int(prediction_class),
            'confidence': confidence,
            'probabilities': probabilities,
            'heart_rate': {
                'mean_bpm': float(hr_metrics.get('hr_mean', 0)),
                'std_bpm': float(hr_metrics.get('hr_std', 0)),
                'min_bpm': float(hr_metrics.get('hr_min', 0)),
                'max_bpm': float(hr_metrics.get('hr_max', 0)),
                'rmssd_ms': float(hr_metrics.get('rmssd', 0)),
                'sdnn_ms': float(hr_metrics.get('sdnn', 0))
            },
            'features_extracted': len(feature_vector[0])
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/extract_features', methods=['POST'])
def extract_features_endpoint():
    """
    Extract features from PPG signal without prediction.
    
    Expected JSON input:
    {
        "ppg": [array of PPG values]
    }
    
    Returns:
    {
        "features": {...},
        "heart_rate": {...}
    }
    """
    try:
        # Parse input
        data = request.get_json()
        
        if 'ppg' not in data:
            return jsonify({'error': 'PPG signal is required'}), 400
        
        ppg_signal = np.array(data['ppg'])
        
        # Validate PPG signal
        if len(ppg_signal) < 64:
            return jsonify({'error': 'PPG signal too short (minimum 64 samples)'}), 400
        
        # Apply bandpass filter to remove noise
        ppg_filtered = bp_feature_extractor.bandpass_filter(ppg_signal)
        
        # Extract heart rate metrics
        hr_metrics = bp_hr_detector.detect_heart_rate(ppg_filtered)
        
        # Extract all features
        features = bp_feature_extractor.extract_all_features(
            ppg=ppg_filtered,
            hr_metrics=hr_metrics
        )
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            return obj
        
        features = convert_to_native(features)
        hr_metrics = convert_to_native(hr_metrics)
        
        return jsonify({
            'features': features,
            'heart_rate': hr_metrics,
            'num_features': len(features)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/info', methods=['GET'])
def info():
    """Get service information."""
    return jsonify({
        'service': 'Combined Blood Pressure & Stress Detection API',
        'version': '2.0.0',
        'models': {
            'blood_pressure': {
                'algorithm': 'Gradient Boosting',
                'classes': 6,
                'class_labels': ['Hypotension', 'Normal', 'Elevated', 
                               'Hypertension Stage 1', 'Hypertension Stage 2', 'Hypertensive Crisis'],
                'features': 17,
                'sampling_rate': '100 Hz',
                'input_samples': 1000
            },
            'stress': {
                'algorithm': 'Gradient Boosting',
                'classes': 2,
                'class_labels': ['Baseline', 'Stress'],
                'features': 21,
                'sampling_rate': {'ppg': '64 Hz', 'temperature': '4 Hz'},
                'input_samples': {'ppg': 640, 'temperature': 40}
            }
        },
        'endpoints': {
            '/health': 'Health check',
            '/predict_bp': 'Predict blood pressure class (requires: ppg)',
            '/predict_stress': 'Predict stress level (requires: ppg, temperature)',
            '/extract_features': 'Extract features from PPG only',
            '/info': 'Service information'
        }
    })

if __name__ == '__main__':
    load_models()
    port = int(os.environ.get('PORT', 7860))  # Hugging Face uses port 7860
    app.run(host='0.0.0.0', port=port, debug=False)
