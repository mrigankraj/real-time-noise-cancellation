from src.core.audio_interface import AudioInterface
from src.core.processors import NoiseProcessor
from src.models.voice_detection import VoiceDetector
import numpy as np
import time

def run():
    # Initialize components
    audio = AudioInterface()
    processor = NoiseProcessor(mode='single_speaker')
    detector = VoiceDetector()
    
    # Initialize output buffer
    output_buffer = []
    
    try:
        print("Starting single speaker noise cancellation...")
        audio.start_stream()
        
        start_time = time.time()
        
        # Process audio stream
        for processed_chunk in audio.process_stream(processor.process_chunk):
            # Check processing latency
            latency = (time.time() - start_time) * 1000  # ms
            if latency > 100:
                print(f"Warning: Processing latency ({latency:.1f}ms) exceeds target")
            
            # Detect voice presence
            has_voice, confidence = detector.detect_voice(processed_chunk)
            
            if has_voice:
                print(f"Voice detected (confidence: {confidence:.2f})")
                output_buffer.append(processed_chunk)
            
            start_time = time.time()
            
    except KeyboardInterrupt:
        print("\nStopping recording...")
    finally:
        # Stop stream and save output
        audio.stop_stream()
        if output_buffer:
            output_data = np.concatenate(output_buffer)
            audio.save_audio("output_single_speaker.wav", output_data)
            print("Saved processed audio to output_single_speaker.wav")

if __name__ == "__main__":
    run()
