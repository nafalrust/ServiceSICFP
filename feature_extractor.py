"""
PPG Feature Extractor for Blood Pressure Classification
Extracts time-domain, frequency-domain, and statistical features from PPG signals
"""

import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis


class PPGFeatureExtractor:
    """Extract comprehensive features from PPG signals."""
    
    def __init__(self, fs=100):
        """
        Initialize feature extractor.
        
        Args:
            fs: Sampling frequency in Hz (default: 100)
        """
        self.fs = fs
    
    def bandpass_filter(self, data, lowcut=0.5, highcut=8.0, order=4):
        """
        Apply bandpass filter to remove noise outside PPG frequency range.
        
        Args:
            data: Raw PPG signal
            lowcut: Low cutoff frequency (Hz) - removes baseline drift
            highcut: High cutoff frequency (Hz) - removes high-frequency noise
            order: Filter order
        
        Returns:
            Filtered PPG signal
        """
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Design bandpass filter
        b, a = signal.butter(order, [low, high], btype='band')
        
        # Apply zero-phase filter (forward and backward)
        filtered_data = signal.filtfilt(b, a, data)
        
        return filtered_data
    
    def extract_all_features(self, ppg, hr_metrics):
        """
        Extract all features from PPG signal only (no demographics).
        
        Args:
            ppg: PPG signal array
            hr_metrics: Dictionary with heart rate metrics from HeartRateDetector
        
        Returns:
            Dictionary with all extracted features (17 features)
        """
        ppg = np.asarray(ppg).flatten()
        
        # Apply bandpass filter to denoise signal
        ppg_filtered = self.bandpass_filter(ppg, lowcut=0.5, highcut=8.0, order=4)
        
        features = {}
        
        # 1. Time Domain Features (9 features)
        time_features = self._extract_time_domain(ppg_filtered)
        features.update(time_features)
        
        # 2. Peak-Based Features (6 features from hr_metrics)
        features['num_peaks'] = hr_metrics.get('num_peaks', 0)
        features['mean_peak_amplitude'] = hr_metrics.get('mean_peak_amplitude', 0)
        features['std_peak_amplitude'] = hr_metrics.get('std_peak_amplitude', 0)
        features['mean_peak_interval'] = hr_metrics.get('mean_peak_interval', 0)
        features['std_peak_interval'] = hr_metrics.get('std_peak_interval', 0)
        features['heart_rate_est'] = hr_metrics.get('hr_mean', 0)  # Use hr_mean as estimate
        
        # 3. Derivative Features (2 features)
        deriv_features = self._extract_derivative_features(ppg_filtered)
        features.update(deriv_features)
        
        return features
    
    def _extract_time_domain(self, signal_data):
        """Extract time domain features (9 features)."""
        features = {}
        
        mean = np.mean(signal_data)
        std = np.std(signal_data)
        
        features['mean'] = mean
        features['std'] = std
        features['min'] = np.min(signal_data)
        features['max'] = np.max(signal_data)
        features['median'] = np.median(signal_data)
        features['range'] = features['max'] - features['min']
        features['rms'] = np.sqrt(np.mean(signal_data**2))
        features['skewness'] = np.mean(((signal_data - mean) / std)**3) if std > 0 else 0
        features['kurtosis'] = np.mean(((signal_data - mean) / std)**4) if std > 0 else 0
        
        return features
    
    def _extract_frequency_simple(self, signal_data):
        """Extract simplified frequency features (2 features)."""
        features = {}
        
        # Simple FFT-based features
        fft_vals = np.abs(np.fft.fft(signal_data))
        
        # Only positive frequencies
        fft_positive = fft_vals[:len(fft_vals)//2]
        
        features['freq_mean'] = np.mean(fft_positive)
        features['freq_std'] = np.std(fft_positive)
        
        return features
    
    def _extract_derivative_features(self, signal_data):
        """Extract derivative features (2 features)."""
        features = {}
        
        # First and second derivatives
        first_derivative = np.diff(signal_data)
        second_derivative = np.diff(first_derivative)
        
        features['mean_first_derivative'] = np.mean(np.abs(first_derivative))
        features['mean_second_derivative'] = np.mean(np.abs(second_derivative))
        
        return features
    
    def features_to_vector(self, features):
        """
        Convert feature dictionary to ordered vector for model prediction.
        
        Args:
            features: Dictionary of extracted features
        
        Returns:
            List of feature values in consistent order (17 features total)
        """
        # Define feature order (must match training order)
        feature_order = [
            # Time domain (9)
            'mean', 'std', 'min', 'max', 'median', 'range', 'rms', 'skewness', 'kurtosis',
            
            # Peak-based (6)
            'num_peaks', 'mean_peak_amplitude', 'std_peak_amplitude',
            'mean_peak_interval', 'std_peak_interval', 'heart_rate_est',
            
            # Derivative (2)
            'mean_first_derivative', 'mean_second_derivative'
        ]
        
        # Extract values in order
        feature_vector = []
        for feature_name in feature_order:
            value = features.get(feature_name, 0)
            feature_vector.append(float(value))
        
        return feature_vector
    
    def get_feature_names(self):
        """Get ordered list of feature names (17 features)."""
        return [
            # Time domain (9)
            'mean', 'std', 'min', 'max', 'median', 'range', 'rms', 'skewness', 'kurtosis',
            
            # Peak-based (6)
            'num_peaks', 'mean_peak_amplitude', 'std_peak_amplitude',
            'mean_peak_interval', 'std_peak_interval', 'heart_rate_est',
            
            # Derivative (2)
            'mean_first_derivative', 'mean_second_derivative'
        ]
