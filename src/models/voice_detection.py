import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class VoiceDetector(nn.Module):
    def __init__(self, n_fft: int = 2048, 
                 hop_length: int = 512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Simple CNN for voice detection
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    
    def detect_voice(self, audio_chunk: np.ndarray) -> Tuple[bool, float]:
        """Detect if voice is present in audio chunk"""
        # Convert to spectrogram
        spec = torch.stft(
            torch.from_numpy(audio_chunk).float(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft),
            return_complex=True
        )
        
        # Prepare input
        mag_spec = torch.abs(spec).unsqueeze(0).unsqueeze(0)
        
        # Get prediction
        with torch.no_grad():
            pred = self.forward(mag_spec)
            
        threshold = 0.5
        return bool(pred > threshold), float(pred)
    
    def save_model(self, path: str) -> None:
        """Save model weights"""
        torch.save(self.state_dict(), path)
    
    def load_model(self, path: str) -> None:
        """Load model weights"""
        self.load_state_dict(torch.load(path))
