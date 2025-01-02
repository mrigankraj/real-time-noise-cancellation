import pytest
import numpy as np
from src.core.processors import NoiseProcessor

def test_noise_processor_initialization():
    processor = NoiseProcessor(mode='single_speaker')
    assert processor.mode == 'single_speaker'
    assert processor.noise_print is None

def test_noise_estimation():
    processor = NoiseProcessor()
    test_chunk = np.random.rand(8820)
    noise_estimate = processor.estimate_noise(test_chunk)
    assert noise_estimate is not None
    assert len(noise_estimate) > 0

def test_spectral_subtraction():
    processor = NoiseProcessor()
    test_chunk = np.random.rand(8820)
    noise_estimate = processor.estimate_noise(test_chunk)
    cleaned = processor.apply_spectral_subtraction(test_chunk, noise_estimate)
    assert cleaned is not None
    assert len(cleaned) == len(test_chunk)

def test_single_speaker_processing():
    processor = NoiseProcessor(mode='single_speaker')
    test_chunk = np.random.rand(8820)
    processed = processor._process_single_speaker(test_chunk)
    assert len(processed) == len(test_chunk)
    assert np.max(np.abs(processed)) <= 1.0

def test_multi_speaker_processing():
    processor = NoiseProcessor(mode='multi_speaker')
    test_chunk = np.random.rand(8820)
    processed = processor._process_multi_speaker(test_chunk)
    assert len(processed) == len(test_chunk)
    assert np.max(np.abs(processed)) <= 1.0
