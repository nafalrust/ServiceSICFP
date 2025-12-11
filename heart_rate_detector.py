"""
Heart Rate Detection from PPG Signal
Multiple peak detection methods: LMM, FDM, BGM
"""

import numpy as np
from scipy.signal import find_peaks, butter, filtfilt


class HeartRateDetector:
    """Detect heart rate and HRV metrics from PPG signals."""
    
    def __init__(self, fs=100):
        """
        Initialize heart rate detector.
        
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
        b, a = butter(order, [low, high], btype='band')
        
        # Apply zero-phase filter (forward and backward)
        filtered_data = filtfilt(b, a, data)
        
        return filtered_data
    
    def detect_heart_rate(self, signal_data):
        """
        Detect peaks and calculate heart rate metrics.
        
        Args:
            signal_data: PPG signal array
        
        Returns:
            Dictionary with heart rate metrics:
                - hr_mean: Mean heart rate in BPM
                - hr_std: Standard deviation of heart rate
                - hr_min: Minimum heart rate
                - hr_max: Maximum heart rate
                - rmssd: Root mean square of successive differences (HRV metric)
                - sdnn: Standard deviation of NN intervals (HRV metric)
                - num_peaks: Number of detected peaks
        """
        signal_data = np.asarray(signal_data).flatten()
        
        # Apply bandpass filter to denoise signal before peak detection
        signal_filtered = self.bandpass_filter(signal_data, lowcut=0.5, highcut=8.0, order=4)
        
        # Detect peaks using Local Maxima Method (LMM)
        peaks = self._detect_peaks_lmm(signal_filtered)
        
        # Calculate heart rate metrics
        metrics = self._calculate_hr_metrics(peaks, signal_filtered)
        
        return metrics
    
    def _detect_peaks_lmm(self, signal_data):
        """
        Local Maxima Method (LMM) for peak detection.
        
        Args:
            signal_data: PPG signal array
        
        Returns:
            Array of peak indices
        """
        # Parameters
        min_distance = int(0.5 * self.fs)  # Minimum 0.5s between peaks (max 120 bpm)
        
        # Normalize signal
        signal_normalized = (signal_data - np.mean(signal_data)) / (np.std(signal_data) + 1e-8)
        
        # Find peaks
        try:
            peaks, properties = find_peaks(
                signal_normalized,
                distance=min_distance,
                prominence=0.1,
                height=0
            )
            return peaks
        except:
            return np.array([])
    
    def _detect_peaks_fdm(self, signal_data):
        """
        First Derivative Method (FDM) for peak detection.
        
        Args:
            signal_data: PPG signal array
        
        Returns:
            Array of peak indices
        """
        # Calculate first derivative
        first_derivative = np.diff(signal_data)
        
        # Find zero crossings where derivative changes from positive to negative
        zero_crossings = np.where(np.diff(np.sign(first_derivative)) < 0)[0]
        
        # Filter peaks that are above median
        valid_peaks = zero_crossings[signal_data[zero_crossings] > np.median(signal_data)]
        
        return valid_peaks
    
    def _detect_peaks_bgm(self, signal_data):
        """
        Second Derivative Method / Beat Gradient Method (BGM) for peak detection.
        
        Args:
            signal_data: PPG signal array
        
        Returns:
            Array of peak indices
        """
        # Calculate second derivative
        first_derivative = np.diff(signal_data)
        second_derivative = np.diff(first_derivative)
        
        # Find zero crossings in second derivative
        zero_crossings = np.where(np.diff(np.sign(second_derivative)) != 0)[0]
        
        # Filter by signal amplitude
        valid_peaks = zero_crossings[signal_data[zero_crossings] > np.mean(signal_data)]
        
        return valid_peaks
    
    def _calculate_hr_metrics(self, peaks, signal_data):
        """
        Calculate heart rate and HRV metrics from detected peaks.
        
        Args:
            peaks: Array of peak indices
            signal_data: Original PPG signal
        
        Returns:
            Dictionary with HR and HRV metrics + peak amplitudes
        """
        metrics = {
            'num_peaks': len(peaks),
            'hr_mean': 0,
            'hr_std': 0,
            'hr_min': 0,
            'hr_max': 0,
            'rmssd': 0,
            'sdnn': 0,
            'mean_peak_amplitude': 0,
            'std_peak_amplitude': 0,
            'mean_peak_interval': 0,
            'std_peak_interval': 0
        }
        
        if len(peaks) < 2:
            return metrics
        
        # Peak amplitudes
        peak_amplitudes = signal_data[peaks]
        metrics['mean_peak_amplitude'] = np.mean(peak_amplitudes)
        metrics['std_peak_amplitude'] = np.std(peak_amplitudes)
        
        # Calculate inter-beat intervals (IBI) in seconds
        peak_intervals = np.diff(peaks) / self.fs
        
        # Calculate heart rates in BPM
        heart_rates = 60 / peak_intervals
        
        # Filter unrealistic heart rates (40-200 BPM)
        valid_mask = (heart_rates >= 40) & (heart_rates <= 200)
        heart_rates_valid = heart_rates[valid_mask]
        peak_intervals_valid = peak_intervals[:-1][valid_mask]  # Adjust length
        
        if len(heart_rates_valid) == 0:
            return metrics
        
        # Heart Rate metrics
        metrics['hr_mean'] = np.mean(heart_rates_valid)
        metrics['hr_std'] = np.std(heart_rates_valid)
        metrics['hr_min'] = np.min(heart_rates_valid)
        metrics['hr_max'] = np.max(heart_rates_valid)
        
        # Peak interval metrics (in seconds, not ms)
        metrics['mean_peak_interval'] = np.mean(peak_intervals)
        metrics['std_peak_interval'] = np.std(peak_intervals)
        
        # HRV metrics (in milliseconds)
        if len(peak_intervals_valid) > 1:
            # RMSSD: Root Mean Square of Successive Differences
            successive_diffs = np.diff(peak_intervals_valid)
            metrics['rmssd'] = np.sqrt(np.mean(successive_diffs**2)) * 1000
            
            # SDNN: Standard Deviation of NN intervals
            metrics['sdnn'] = np.std(peak_intervals_valid) * 1000
        
        return metrics
    
    def visualize_peaks(self, signal_data, peaks=None):
        """
        Generate peak detection visualization data.
        
        Args:
            signal_data: PPG signal array
            peaks: Array of peak indices (if None, will detect automatically)
        
        Returns:
            Dictionary with visualization data
        """
        signal_data = np.asarray(signal_data).flatten()
        
        if peaks is None:
            peaks = self._detect_peaks_lmm(signal_data)
        
        # Time axis
        time = np.arange(len(signal_data)) / self.fs
        
        return {
            'time': time.tolist(),
            'signal': signal_data.tolist(),
            'peak_indices': peaks.tolist(),
            'peak_times': (peaks / self.fs).tolist(),
            'peak_values': signal_data[peaks].tolist()
        }
