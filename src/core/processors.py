import numpy as np
from scipy import signal
from typing import Optional, Tuple
import torch
import torchaudio

class NoiseProcessor:
    def __init__(self, mode: str = 'single_speaker'):
        self.mode = mode
        self.noise_print = None
        self.spectral_floor = 0.002
        
    def estimate_noise(self, audio_chunk: np.ndarray, 
                      window_size: int = 2048) -> np.ndarray:
        """Estimate noise profile from audio chunk"""
        spectrum = np.fft.rfft(audio_chunk, window_size)
        magnitude = np.abs(spectrum)
        noise_estimate = np.minimum(magnitude, 
                                  np.median(magnitude) * 1.5)
        return noise_estimate
    
    def apply_spectral_subtraction(self, 
                                 audio_chunk: np.ndarray,
                                 noise_estimate: np.ndarray) -> np.ndarray:
        """Apply spectral subtraction for noise reduction"""
        spectrum = np.fft.rfft(audio_chunk)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        
        # Subtract noise estimate from magnitude
        magnitude = np.maximum(
            magnitude - noise_estimate * 1.5,
            magnitude * self.spectral_floor
        )
        
        # Reconstruct signal
        spectrum_clean = magnitude * np.exp(1j * phase)
        return np.fft.irfft(spectrum_clean)
    
    def process_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Process audio chunk based on mode"""
        if self.mode == 'single_speaker':
            return self._process_single_speaker(audio_chunk)
        else:
            return self._process_multi_speaker(audio_chunk)
    
    def _process_single_speaker(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Process audio for single speaker scenario"""
        if self.noise_print is None:
            self.noise_print = self.estimate_noise(audio_chunk)
            
        # Update noise estimate
        self.noise_print = 0.95 * self.noise_print + \
                          0.05 * self.estimate_noise(audio_chunk)
        
        return self.apply_spectral_subtraction(audio_chunk, 
                                             self.noise_print)
    
    def _process_multi_speaker(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Process audio for multi-speaker scenario"""
        # Use more conservative noise reduction
        noise_estimate = self.estimate_noise(audio_chunk)
        noise_estimate *= 0.5  # Reduce aggressiveness
        
        return self.apply_spectral_subtraction(audio_chunk, 
                                             noise_estimate)
