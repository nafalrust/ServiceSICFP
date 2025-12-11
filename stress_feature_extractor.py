"""
Stress Feature Extractor
Extracts 21 features from PPG and Temperature signals for stress classification
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq


class StressFeatureExtractor:
    """Extract features from PPG and Temperature signals for stress detection."""
    
    def __init__(self, fs=64):
        """
        Initialize feature extractor.
        
        Args:
            fs: Sampling frequency in Hz (default: 64 for PPG)
        """
        self.fs = fs
    
    def extract_simple_features(self, signal_data):
        """
        Extract basic statistical features (6 features).
        
        Returns:
            dict with mean, std, min, max, median, rms
        """
        features = {}
        features['mean'] = np.mean(signal_data)
        features['std'] = np.std(signal_data)
        features['min'] = np.min(signal_data)
        features['max'] = np.max(signal_data)
        features['median'] = np.median(signal_data)
        features['rms'] = np.sqrt(np.mean(signal_data**2))
        return features
    
    def extract_frequency_simple(self, signal_data):
        """
        Extract simple frequency domain features (3 features).
        
        Returns:
            dict with dominant_freq, freq_energy, freq_mean
        """
        features = {}
        
        # FFT
        fft_vals = fft(signal_data)
        freqs = fftfreq(len(signal_data), 1/self.fs)
        
        # Positive frequencies only
        positive_mask = freqs > 0
        positive_freqs = freqs[positive_mask]
        positive_fft = np.abs(fft_vals[positive_mask])
        
        # Extract features
        features['dominant_freq'] = positive_freqs[np.argmax(positive_fft)] if len(positive_fft) > 0 else 0
        features['freq_energy'] = np.sum(positive_fft**2)
        features['freq_mean'] = np.mean(positive_fft)
        
        return features
    
    def extract_all_features(self, ppg, temp, hr_metrics):
        """
        Extract all 21 features for stress classification.
        
        Args:
            ppg: PPG signal array
            temp: Temperature signal array
            hr_metrics: Dictionary with heart rate metrics from HeartRateDetector
        
        Returns:
            Dictionary with 21 features
        """
        ppg = np.asarray(ppg).flatten()
        temp = np.asarray(temp).flatten()
        
        features = {}
        
        # PPG features: 6 statistical + 3 frequency = 9 features
        ppg_stats = self.extract_simple_features(ppg)
        ppg_freq = self.extract_frequency_simple(ppg)
        
        for key, val in ppg_stats.items():
            features[f'ppg_{key}'] = val
        for key, val in ppg_freq.items():
            features[f'ppg_{key}'] = val
        
        # Temperature features: 6 statistical features
        temp_stats = self.extract_simple_features(temp)
        for key, val in temp_stats.items():
            features[f'temp_{key}'] = val
        
        # Heart rate metrics: 6 features (skip num_peaks)
        for key, val in hr_metrics.items():
            if key not in ['peaks', 'num_peaks', 'mean_peak_amplitude', 'std_peak_amplitude', 
                          'mean_peak_interval', 'std_peak_interval']:
                features[f'hr_{key}'] = val
        
        return features
    
    def features_to_vector(self, features):
        """
        Convert feature dictionary to ordered vector for model prediction.
        Order must match training data.
        
        Args:
            features: Dictionary of extracted features
        
        Returns:
            List of feature values in consistent order (21 features)
        """
        # Define feature order (must match training order)
        feature_order = [
            # PPG statistical (6)
            'ppg_mean', 'ppg_std', 'ppg_min', 'ppg_max', 'ppg_median', 'ppg_rms',
            
            # PPG frequency (3)
            'ppg_dominant_freq', 'ppg_freq_energy', 'ppg_freq_mean',
            
            # Temperature statistical (6)
            'temp_mean', 'temp_std', 'temp_min', 'temp_max', 'temp_median', 'temp_rms',
            
            # Heart rate (6)
            'hr_hr_mean', 'hr_hr_std', 'hr_hr_min', 'hr_hr_max', 'hr_rmssd', 'hr_sdnn'
        ]
        
        # Extract values in order
        feature_vector = []
        for feature_name in feature_order:
            value = features.get(feature_name, 0)
            feature_vector.append(float(value))
        
        return feature_vector
    
    def get_feature_names(self):
        """Get ordered list of feature names (21 features)."""
        return [
            # PPG statistical (6)
            'ppg_mean', 'ppg_std', 'ppg_min', 'ppg_max', 'ppg_median', 'ppg_rms',
            
            # PPG frequency (3)
            'ppg_dominant_freq', 'ppg_freq_energy', 'ppg_freq_mean',
            
            # Temperature statistical (6)
            'temp_mean', 'temp_std', 'temp_min', 'temp_max', 'temp_median', 'temp_rms',
            
            # Heart rate (6)
            'hr_hr_mean', 'hr_hr_std', 'hr_hr_min', 'hr_hr_max', 'hr_rmssd', 'hr_sdnn'
        ]
