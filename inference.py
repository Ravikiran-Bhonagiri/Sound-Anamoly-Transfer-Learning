import os
import numpy as np
import sounddevice as sd
import tflite_runtime.interpreter as tflite
import time

# Configuration
import queue

import argparse
import sys

# Configuration
# MODEL_PATH will be set via argument
CLASSES = ['on_state', 'off_state', 'solid_state', 'soft_state']
SAMPLE_RATE = 48000
TARGET_SR = 16000
BUFFER_DURATION = 1.0 # seconds
BUFFER_SIZE = int(TARGET_SR * BUFFER_DURATION) # 16000 samples
BLOCK_SIZE = 4800 # Process in chunks of 0.1s (4800 samples at 48k)

# Thread-safe queue for audio blocks
audio_queue = queue.Queue()

def process_audio_chunk(input_data):
    """
    Downsamples 48kHz input to 16kHz.
    """
    # Simple slicing: take every 3rd sample (48000 / 3 = 16000)
    return input_data[::3]

def audio_callback(indata, frames, time, status):
    """Callback for sounddevice. Runs in a separate thread."""
    if status:
        print(status)
    
    # We just put the raw data into the queue to minimize blocking
    # We make a copy to ensure thread safety
    audio_queue.put(indata.copy())

def run_inference(interpreter, input_details, output_details, input_data):
    """Runs inference on the provided input data."""
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0] # Probability vector

def main():
    parser = argparse.ArgumentParser(description='Sound Anomaly Detection Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to .tflite model file')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        sys.exit(1)

    print(f"Loading model from {args.model}...")
    interpreter = tflite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Model loaded. Input shape: {input_details[0]['shape']}")
    
    # Local rolling buffer
    audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
    
    print("Starting audio stream...")
    
    # Start stream
    with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=1, callback=audio_callback):
        print("Listening... Press Ctrl+C to stop.")
        try:
            while True:
                # Get block from queue (blocking wait)
                # This effectively synchronizes the loop with the audio stream
                raw_block = audio_queue.get()
                
                # Flatten and downsample
                data = raw_block.flatten()
                downsampled = process_audio_chunk(data)
                
                # Update rolling buffer
                n_new = len(downsampled)
                audio_buffer = np.roll(audio_buffer, -n_new)
                audio_buffer[-n_new:] = downsampled
                
                # Run inference
                start_time = time.time()
                
                # Prepare input (float32)
                input_data = audio_buffer.astype(np.float32)
                
                probs = run_inference(interpreter, input_details, output_details, input_data)
                prediction_idx = np.argmax(probs)
                confidence = probs[prediction_idx]
                label = CLASSES[prediction_idx]
                
                end_time = time.time()
                inference_time = (end_time - start_time) * 1000 # ms
                
                # Print status
                print(f"State: {label: <15} (Conf: {confidence:.2f}) | Inf: {inference_time:.1f}ms | Queue: {audio_queue.qsize()}", end='\r')
                
        except KeyboardInterrupt:
            print("\nStopped.")

if __name__ == "__main__":
    main()
