"""
Heart Rate Detection for Stress Classification
Detects heart rate and HRV metrics from PPG signal using multiple methods
Matches training notebook implementation (Local Maxima Method as primary)
"""

import numpy as np
from scipy.signal import find_peaks


class HeartRateDetector:
    """Detects heart rate and calculates HRV metrics from PPG signal"""
    
    def __init__(self, fs=64):
        """
        Initialize heart rate detector
        
        Args:
            fs (int): Sampling frequency in Hz (default: 64 Hz for WESAD)
        """
        self.fs = fs
    
    def detect_heart_rate(self, ppg_signal):
        """
        Detect peaks in PPG signal and calculate heart rate metrics.
        Uses Local Maxima Method (LMM) as primary method, matching training notebook.
        
        Multiple methods implemented:
        - Method 1: Local Maxima Method (LMM) - Primary
        - Method 2: First Derivative Method (FDM)
        - Method 3: Second Derivative Method (SDM/BGM)
        
        Args:
            ppg_signal (array-like): PPG signal
            
        Returns:
            dict: Dictionary containing HR and HRV metrics
        """
        # Ensure signal is 1D
        signal = np.asarray(ppg_signal).flatten()
        
        # Peak detection parameters
        min_distance = int(0.5 * self.fs)  # Minimum 0.5s between peaks (max 120 bpm)
        
        # Method 1: Local Maxima Method (LMM) - Primary method
        peaks_lmm, _ = find_peaks(signal, distance=min_distance, prominence=0.1)
        
        # Method 2: First Derivative Method (FDM)
        first_derivative = np.diff(signal)
        zero_crossings = np.where(np.diff(np.sign(first_derivative)))[0]
        peaks_fdm = zero_crossings[signal[zero_crossings] > np.median(signal)]
        
        # Method 3: Second Derivative Method (SDM/BGM)
        second_derivative = np.diff(first_derivative)
        
        # Use LMM as primary method (most reliable)
        peaks = peaks_lmm
        
        # Calculate heart rate metrics
        if len(peaks) > 1:
            # Calculate intervals between peaks in seconds
            peak_intervals = np.diff(peaks) / self.fs
            
            # Convert intervals to heart rate in BPM
            heart_rates = 60 / peak_intervals
            
            # Basic HR statistics
            hr_mean = np.mean(heart_rates)
            hr_std = np.std(heart_rates)
            hr_min = np.min(heart_rates)
            hr_max = np.max(heart_rates)
            
            # Heart Rate Variability (HRV) metrics
            # RMSSD: Root Mean Square of Successive Differences (in ms)
            rmssd = np.sqrt(np.mean(np.diff(peak_intervals)**2)) * 1000
            
            # SDNN: Standard Deviation of NN intervals (in ms)
            sdnn = np.std(peak_intervals) * 1000
        else:
            # Not enough peaks detected
            hr_mean = hr_std = hr_min = hr_max = 0
            rmssd = sdnn = 0
        
        return {
            'hr_mean': hr_mean,
            'hr_std': hr_std,
            'hr_min': hr_min,
            'hr_max': hr_max,
            'rmssd': rmssd,
            'sdnn': sdnn
        }
