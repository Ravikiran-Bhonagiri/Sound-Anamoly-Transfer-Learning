import numpy as np
import tflite_runtime.interpreter as tflite
import wave
import sys
import os
import time

# Configuration
MODEL_PATH = 'artifacts/sound_classifier.tflite'
CLASSES = ['on_state', 'off_state', 'solid_state', 'soft_state']
TARGET_SR = 16000
INPUT_LENGTH = 16000

def process_audio(file_path):
    with wave.open(file_path, 'rb') as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        bytes_data = wf.readframes(n_frames)
        
        # Convert to numpy array (assuming 16-bit PCM)
        wav_data = np.frombuffer(bytes_data, dtype=np.int16)
        
        # Normalize to float [-1, 1]
        wav_data = wav_data.astype(np.float32) / 32768.0
        
        # Resample if necessary
        if sr != TARGET_SR:
            # Simple downsampling if integer ratio
            if sr == 48000:
                 wav_data = wav_data[::3]
            else:
                # Naive resampling for other rates (linear interpolation)
                # This is a test script, so simple is okay
                indices = np.arange(0, len(wav_data), sr / TARGET_SR)
                indices = indices[indices < len(wav_data)].astype(int)
                wav_data = wav_data[indices]
                
    return wav_data

def main():
    if len(sys.argv) < 3:
        print("Usage: python test_docker_inference.py <wav_file> <model_path>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    model_path = sys.argv[2]
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)
        
    print(f"Testing file: {file_path}")
    print(f"Using model: {model_path}")
    
    # Load Model
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"Input Details: {input_details}")
    print(f"Output Details: {output_details}")
    
    # Resize input to match chunk size
    input_index = input_details[0]['index']
    interpreter.resize_tensor_input(input_index, [INPUT_LENGTH])
    interpreter.allocate_tensors()
    
    # Process Audio
    audio_data = process_audio(file_path)
    
    # Run inference on chunks
    chunk_size = INPUT_LENGTH
    print(f"Audio length: {len(audio_data)} samples. Processing {len(audio_data)//chunk_size} chunks...")
    
    for start in range(0, len(audio_data) - chunk_size + 1, chunk_size):
        chunk = audio_data[start:start+chunk_size]
        
        # Set input tensor (1D array)
        interpreter.set_tensor(input_index, chunk)
        
        # Measure inference time
        start_time = time.time()
        interpreter.invoke()
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000 # ms
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        probs = output_data[0]
        prediction_idx = np.argmax(probs)
        label = CLASSES[prediction_idx]
        confidence = probs[prediction_idx]
        
        print(f"Time {start/TARGET_SR:.2f}s: {label} ({confidence:.2f}) | Chunk: {chunk.shape} | Inference: {inference_time:.2f}ms")

if __name__ == "__main__":
    main()
