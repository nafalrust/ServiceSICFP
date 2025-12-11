#!/usr/bin/env python3
"""
Test script for Combined AI Service (Blood Pressure + Stress Detection)
Tests all endpoints with sample data
"""

import requests
import numpy as np
import json


BASE_URL = "http://localhost:7860"


def test_health():
    """Test health endpoint"""
    print("\n=== Testing /health endpoint ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✓ Health check passed")


def test_info():
    """Test info endpoint"""
    print("\n=== Testing /info endpoint ===")
    response = requests.get(f"{BASE_URL}/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✓ Info endpoint passed")


def generate_bp_signal():
    """Generate sample PPG signal for BP classification"""
    # PPG signal: 10 seconds @ 100 Hz = 1000 samples
    t = np.linspace(0, 10, 1000)
    heart_rate = 72 / 60  # 72 BPM in Hz
    ppg_signal = np.sin(2 * np.pi * heart_rate * t) + np.random.normal(0, 0.1, 1000)
    
    return ppg_signal.tolist()


def generate_stress_signals():
    """Generate sample PPG and temperature signals for stress detection"""
    # PPG signal: 10 seconds @ 64 Hz = 640 samples
    t_ppg = np.linspace(0, 10, 640)
    heart_rate = 75 / 60  # 75 BPM in Hz
    ppg_signal = np.sin(2 * np.pi * heart_rate * t_ppg) + np.random.normal(0, 0.1, 640)
    
    # Temperature signal: 10 seconds @ 4 Hz = 40 samples
    temp_signal = 36.5 + np.random.normal(0, 0.2, 40)
    
    return ppg_signal.tolist(), temp_signal.tolist()


def test_predict_bp_normal():
    """Test BP prediction with normal signal"""
    print("\n=== Testing /predict_bp endpoint (Normal BP) ===")
    ppg = generate_bp_signal()
    
    data = {"ppg": ppg}
    
    response = requests.post(f"{BASE_URL}/predict_bp", json=data)
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    
    assert response.status_code == 200
    assert "prediction" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert "heart_rate" in result
    assert len(result["probabilities"]) == 6
    print(f"✓ Prediction: {result['prediction']} (confidence: {result['confidence']:.2%})")


def test_predict_stress_baseline():
    """Test stress prediction with baseline (no stress) signal"""
    print("\n=== Testing /predict_stress endpoint (Baseline) ===")
    ppg, temp = generate_stress_signals()
    
    data = {
        "ppg": ppg,
        "temperature": temp
    }
    
    response = requests.post(f"{BASE_URL}/predict_stress", json=data)
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    
    assert response.status_code == 200
    assert "prediction" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert "heart_rate" in result
    assert len(result["probabilities"]) == 2
    print(f"✓ Prediction: {result['prediction']} (confidence: {result['confidence']:.2%})")


def test_predict_stress_elevated():
    """Test stress prediction with elevated HR and temp (stress indicators)"""
    print("\n=== Testing /predict_stress endpoint (Stress) ===")
    
    # Simulate stress: higher heart rate and temperature
    t_ppg = np.linspace(0, 10, 640)
    heart_rate = 95 / 60  # Elevated HR
    ppg_signal = np.sin(2 * np.pi * heart_rate * t_ppg) + np.random.normal(0, 0.15, 640)
    
    temp_signal = 37.2 + np.random.normal(0, 0.3, 40)  # Elevated temperature
    
    data = {
        "ppg": ppg_signal.tolist(),
        "temperature": temp_signal.tolist()
    }
    
    response = requests.post(f"{BASE_URL}/predict_stress", json=data)
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    
    assert response.status_code == 200
    print(f"✓ Prediction: {result['prediction']} (confidence: {result['confidence']:.2%})")


def test_extract_features():
    """Test feature extraction endpoint"""
    print("\n=== Testing /extract_features endpoint ===")
    ppg = generate_bp_signal()
    
    data = {"ppg": ppg}
    
    response = requests.post(f"{BASE_URL}/extract_features", json=data)
    print(f"Status Code: {response.status_code}")
    result = response.json()
    
    print(f"Number of features extracted: {result['num_features']}")
    print(f"\nSample features:")
    for i, (key, value) in enumerate(list(result['features'].items())[:5]):
        print(f"  {key}: {value:.4f}")
    
    print(f"\nHeart Rate Metrics:")
    for key, value in result['heart_rate'].items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    assert response.status_code == 200
    assert result['num_features'] == 17
    print("✓ Feature extraction passed")


def test_invalid_bp_signal():
    """Test error handling for invalid BP signal"""
    print("\n=== Testing invalid PPG length (BP) ===")
    ppg = [0.5] * 50  # Too short
    
    data = {"ppg": ppg}
    
    response = requests.post(f"{BASE_URL}/predict_bp", json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 400
    print("✓ Error handling passed")


def test_missing_temperature():
    """Test error handling for missing temperature in stress prediction"""
    print("\n=== Testing missing temperature ===")
    ppg = generate_bp_signal()
    
    data = {"ppg": ppg}
    
    response = requests.post(f"{BASE_URL}/predict_stress", json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 400
    print("✓ Error handling passed")


def run_all_tests():
    """Run all test cases"""
    print("=" * 70)
    print("COMBINED AI SERVICE - TEST SUITE")
    print("Blood Pressure & Stress Detection")
    print("=" * 70)
    
    try:
        test_health()
        test_info()
        test_predict_bp_normal()
        test_predict_stress_baseline()
        test_predict_stress_elevated()
        test_extract_features()
        test_invalid_bp_signal()
        test_missing_temperature()
        
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Cannot connect to service at {BASE_URL}")
        print("Make sure the service is running: python app.py")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(run_all_tests())
