import numpy as np
import sounddevice as sd
import wave
from typing import Optional, Tuple, Generator
import queue
import threading

class AudioInterface:
    def __init__(self, 
                 sample_rate: int = 44100,
                 chunk_size: int = 8820,  # 200ms chunks at 44.1kHz
                 channels: int = 1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
    def start_stream(self) -> None:
        """Start audio input stream"""
        self.is_recording = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self._audio_callback,
            blocksize=self.chunk_size
        )
        self.stream.start()

    def stop_stream(self) -> None:
        """Stop audio input stream"""
        self.is_recording = False
        self.stream.stop()
        self.stream.close()

    def _audio_callback(self, indata: np.ndarray, frames: int, 
                       time_info: dict, status: sd.CallbackFlags) -> None:
        """Callback for audio stream processing"""
        if status:
            print(f"Stream callback error: {status}")
        self.audio_queue.put(indata.copy())

    def get_audio_chunk(self) -> Optional[np.ndarray]:
        """Get next chunk of audio data"""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None

    def save_audio(self, filename: str, audio_data: np.ndarray) -> None:
        """Save audio data to WAV file"""
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())

    def process_stream(self, processor: callable) -> Generator[np.ndarray, None, None]:
        """Process audio stream with given processor function"""
        while self.is_recording:
            chunk = self.get_audio_chunk()
            if chunk is not None:
                processed_chunk = processor(chunk)
                yield processed_chunk
