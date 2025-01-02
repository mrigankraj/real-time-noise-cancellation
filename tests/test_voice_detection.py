import pytest
import torch
import numpy as np
from src.models.voice_detection import VoiceDetector

def test_voice_detector_initialization():
    detector = VoiceDetector()
    assert isinstance(detector, torch.nn.Module)

def test_forward_pass():
    detector = VoiceDetector()
    test_input = torch.randn(1, 1, 64, 64)
    output = detector.forward(test_input)
    assert output.shape == (1, 1)
    assert 0 <= float(output) <= 1

def test_voice_detection():
    detector = VoiceDetector()
    test_chunk = np.random.rand(8820)
    is_voice, confidence = detector.detect_voice(test_chunk)
    assert isinstance(is_voice, bool)
    assert 0 <= confidence <= 1

def test_model_save_load():
    detector = VoiceDetector()
    detector.save_model("test_model.pt")
    
    new_detector = VoiceDetector()
    new_detector.load_model("test_model.pt")
    
    test_input = torch.randn(1, 1, 64, 64)
    assert torch.allclose(
        detector(test_input),
        new_detector(test_input)
    )
