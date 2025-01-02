import pytest
import numpy as np
from src.core.audio_interface import AudioInterface

def test_audio_interface_initialization():
    audio = AudioInterface()
    assert audio.sample_rate == 44100
    assert audio.chunk_size == 8820
    assert audio.channels == 1

def test_audio_stream_start_stop():
    audio = AudioInterface()
    audio.start_stream()
    assert audio.is_recording == True
    audio.stop_stream()
    assert audio.is_recording == False

def test_audio_chunk_processing():
    audio = AudioInterface()
    test_chunk = np.random.rand(8820)
    
    def mock_processor(chunk):
        return chunk * 0.5
    
    audio.start_stream()
    processed = next(audio.process_stream(mock_processor))
    audio.stop_stream()
    
    assert processed is not None
    assert len(processed) == 8820

def test_wav_file_saving():
    audio = AudioInterface()
    test_data = np.random.rand(8820).astype(np.float32)
    
    audio.save_audio("test_output.wav", test_data)
    # Verify file exists and has correct format
    import wave
    with wave.open("test_output.wav", 'rb') as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 44100
