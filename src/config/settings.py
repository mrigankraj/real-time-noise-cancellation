# Audio processing settings
SAMPLE_RATE = 44100
CHUNK_SIZE = 8820  # 200ms at 44.1kHz
CHANNELS = 1

# FFT settings
N_FFT = 2048
HOP_LENGTH = 512
WINDOW_SIZE = 2048

# Voice detection settings
VOICE_DETECTION_THRESHOLD = 0.5
VOICE_MIN_DURATION = 0.1  # seconds

# Noise reduction settings
NOISE_REDUCTION_STRENGTH = 1.5
SPECTRAL_FLOOR = 0.002
NOISE_UPDATE_RATE = 0.05

# Performance settings
MAX_LATENCY = 100  # milliseconds
BUFFER_SIZE = 10  # chunks

# File settings
OUTPUT_FORMAT = 'wav'
DEFAULT_OUTPUT_PATH = 'output/'
