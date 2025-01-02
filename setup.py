from setuptools import setup, find_packages

setup(
    name="noise-cancellation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "librosa>=0.9.0",
        "sounddevice>=0.4.4",
        "PyAudio>=0.2.11",
        "torch>=1.9.0",
        "torchaudio>=0.9.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Real-time noise cancellation system",
    keywords="audio, noise-cancellation, signal-processing",
    python_requires=">=3.8",
)
