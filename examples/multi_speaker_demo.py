from src.core.audio_interface import AudioInterface
from src.core.processors import NoiseProcessor
from src.models.voice_detection import VoiceDetector
import numpy as np
import time

def run():
    audio = AudioInterface()
    processor = NoiseProcessor(mode='multi_speaker')
    detector = VoiceDetector()
    
    output_buffer = []
    
    try:
        print("Starting multi-speaker noise cancellation...")
        audio.start_stream()
        
        start_time = time.time()
        
        for processed_chunk in audio.process_stream(processor.process_chunk):
            latency = (time.time() - start_time) * 1000
            if latency > 100:
                print(f"Warning: Processing latency ({latency:.1f}ms) exceeds target")
            
            voices_detected = detector.detect_voice(processed_chunk)
            output_buffer.append(processed_chunk)
            
            start_time = time.time()
            
    except KeyboardInterrupt:
        print("\nStopping recording...")
    finally:
        audio.stop_stream()
        if output_buffer:
            output_data = np.concatenate(output_buffer)
            audio.save_audio("output_multi_speaker.wav", output_data)
            print("Saved processed audio to output_multi_speaker.wav")

if __name__ == "__main__":
    run()
