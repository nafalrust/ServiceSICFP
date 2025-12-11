"""
Test the Blood Pressure Classification Service
Run this script to test all endpoints
"""

import requests
import numpy as np
import json

# Service URL
BASE_URL = "http://localhost:7860"

def test_health():
    """Test health check endpoint."""
    print("=" * 70)
    print("Testing /health endpoint...")
    print("=" * 70)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_info():
    """Test info endpoint."""
    print("=" * 70)
    print("Testing /info endpoint...")
    print("=" * 70)
    
    response = requests.get(f"{BASE_URL}/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_extract_features():
    """Test feature extraction endpoint."""
    print("=" * 70)
    print("Testing /extract_features endpoint...")
    print("=" * 70)
    
    # Generate synthetic PPG signal (1000 samples = 10 seconds at 100 Hz)
    t = np.linspace(0, 10, 1000)
    # Simulate realistic PPG: sine wave (heart beats) + noise
    ppg_signal = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(1000)
    
    data = {
        "ppg": ppg_signal.tolist()
    }
    
    response = requests.post(f"{BASE_URL}/extract_features", json=data)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Total features extracted: {result['num_features']}")
        print(f"\nSample features:")
        for i, (key, value) in enumerate(list(result['features'].items())[:10]):
            print(f"  {key}: {value:.4f}")
        print(f"  ... and {len(result['features']) - 10} more")
        
        print(f"\nHeart Rate Metrics:")
        for key, value in result['heart_rate'].items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
    else:
        print(f"Error: {response.text}")
    
    print()

def test_predict():
    """Test prediction endpoint."""
    print("=" * 70)
    print("Testing /predict endpoint...")
    print("=" * 70)
    
    # Generate synthetic PPG signal (1000 samples)
    t = np.linspace(0, 10, 1000)
    ppg_signal = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(1000)
    
    data = {
        "ppg": ppg_signal.tolist()
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nPrediction Results:")
        print(f"  Class: {result['prediction']} (ID: {result['class_id']})")
        if result['confidence']:
            print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Features Extracted: {result['features_extracted']}")
        
        print(f"\nHeart Rate:")
        for key, value in result['heart_rate'].items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        if result['probabilities']:
            print(f"\nClass Probabilities:")
            classes = ['Hypotension', 'Optimal', 'Normal', 'High Normal', 'Hypertension']
            for i, (cls, prob) in enumerate(zip(classes, result['probabilities'])):
                print(f"  {cls}: {prob:.2%}")
    else:
        print(f"Error: {response.text}")
    
    print()

def test_error_handling():
    """Test error handling."""
    print("=" * 70)
    print("Testing error handling...")
    print("=" * 70)
    
    # Test 1: Missing PPG signal
    print("\n1. Missing PPG signal:")
    response = requests.post(f"{BASE_URL}/predict", json={})
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    # Test 2: PPG signal too short
    print("\n2. PPG signal too short:")
    response = requests.post(f"{BASE_URL}/predict", json={"ppg": [1, 2, 3]})
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    print()

def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("BLOOD PRESSURE CLASSIFICATION SERVICE - TEST SUITE")
    print("=" * 70 + "\n")
    
    try:
        test_health()
        test_info()
        test_extract_features()
        test_predict()
        test_error_handling()
        
        print("=" * 70)
        print("✓ All tests completed!")
        print("=" * 70)
        
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Could not connect to service")
        print(f"Make sure the service is running on {BASE_URL}")
        print("\nTo start the service, run:")
        print("  python app.py")
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")

if __name__ == "__main__":
    main()
