# Real-Time Noise Cancellation System

A Python-based real-time noise cancellation system that supports both single-speaker and multi-speaker scenarios.

## Features
- Real-time audio processing with < 100ms latency
- Single speaker mode: Isolates primary speaker while reducing background noise
- Multi-speaker mode: Preserves multiple voices while filtering environmental noise
- Live microphone input processing
- WAV file output of processed audio
- Comprehensive test suite

## Real-Time Noise Cancellation Visualization
![Screenshot (892)](https://github.com/user-attachments/assets/bb46730c-f3e1-479e-bbcc-c666b1f91217)
![Screenshot (896)](https://github.com/user-attachments/assets/ca93ef23-dc7b-4069-91f4-30bbbee1ca67)


## Installation
```bash
pip install -r requirements.txt
python setup.py install
```

## Quick Start
```python
from src.examples import single_speaker_demo, multi_speaker_demo

# For single speaker noise cancellation
single_speaker_demo.run()

# For multi-speaker noise cancellation
multi_speaker_demo.run()
```

## Requirements
- Python 3.8+
- PyAudio
- NumPy
- SciPy
- librosa
- sounddevice

## System Architecture
- Core: Audio interface and processing components
- Models: Neural network-based noise suppression and voice detection
- Config: System settings and parameters
- Tests: Comprehensive test suite
- Examples: Demo applications

## License
MIT License
