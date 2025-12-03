import os
import numpy as np
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite
from scipy.io import wavfile
from scipy import signal
import datetime

# Configuration
MODEL_PATH = 'artifacts/sound_classifier.tflite'
DATA_DIR = 'data'
CLASSES = ['on_state', 'off_state', 'solid_state', 'soft_state']
TARGET_SR = 16000
CHUNK_SIZE = 16000  # 1 second

def load_tflite_model(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    # Resize input to match our chunk size (16000)
    # Get input index
    input_details = interpreter.get_input_details()
    input_index = input_details[0]['index']
    interpreter.resize_tensor_input(input_index, [CHUNK_SIZE])
    interpreter.allocate_tensors()
    return interpreter

def predict_chunk(interpreter, audio_chunk):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Preprocess: Normalize float32
    if audio_chunk.dtype != np.float32:
        # Already normalized in main loop, but safety check
        pass
        
    interpreter.set_tensor(input_details[0]['index'], audio_chunk)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0] # Probabilities

def main():
    print("Starting batch inference...")
    interpreter = load_tflite_model(MODEL_PATH)
    
    log_filename = f"inference_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    total_files = 0
    correct_predictions = 0
    
    with open(log_filename, 'w') as f:
        f.write(f"Batch Inference Log - {datetime.datetime.now()}\n")
        f.write("Filename | True Label | Predicted Label | Confidence | Result\n")
        f.write("-" * 80 + "\n")
        
        for class_name in CLASSES:
            class_dir = os.path.join(DATA_DIR, class_name)
            if not os.path.exists(class_dir):
                continue
                
            print(f"Processing class: {class_name}")
            
            for filename in os.listdir(class_dir):
                if not filename.lower().endswith('.wav'):
                    continue
                    
                filepath = os.path.join(class_dir, filename)
                
                try:
                    sr, wav_data = wavfile.read(filepath)
                    print(f"DEBUG: {filename} - SR: {sr}, Dtype: {wav_data.dtype}, Min: {wav_data.min()}, Max: {wav_data.max()}")
                    
                    # Normalize first
                    if wav_data.dtype == np.int16:
                        wav_data = wav_data / 32768.0
                    elif wav_data.dtype == np.int32:
                        wav_data = wav_data / 2147483648.0
                    elif wav_data.dtype == np.uint8:
                        wav_data = (wav_data - 128) / 128.0
                    
                    wav_data = wav_data.astype(np.float32)
                    
                    # Convert to mono if stereo
                    if len(wav_data.shape) > 1:
                        wav_data = np.mean(wav_data, axis=1)

                    # Resample if needed
                    if sr != TARGET_SR:
                        num_samples = int(len(wav_data) * TARGET_SR / sr)
                        wav_data = signal.resample(wav_data, num_samples)

                    # Process chunks and average predictions
                    chunk_preds = []
                    for start in range(0, len(wav_data) - CHUNK_SIZE + 1, CHUNK_SIZE):
                        chunk = wav_data[start:start+CHUNK_SIZE]
                        pred = predict_chunk(interpreter, chunk)
                        chunk_preds.append(pred)
                        
                    if not chunk_preds:
                        f.write(f"{filename} | {class_name} | SKIPPED (Too short) | - | -\n")
                        continue
                        
                    avg_pred = np.mean(chunk_preds, axis=0)
                    pred_idx = np.argmax(avg_pred)
                    pred_label = CLASSES[pred_idx]
                    confidence = avg_pred[pred_idx]
                    
                    is_correct = (pred_label == class_name)
                    result = "PASS" if is_correct else "FAIL"
                    
                    if is_correct:
                        correct_predictions += 1
                    total_files += 1
                    
                    log_line = f"{filename} | {class_name} | {pred_label} | {confidence:.4f} | {result}\n"
                    f.write(log_line)
                    # print(log_line.strip()) # Optional: print to console
                    
                except Exception as e:
                    f.write(f"{filename} | {class_name} | ERROR: {str(e)} | - | -\n")
                    print(f"Error processing {filename}: {e}")

        accuracy = (correct_predictions / total_files) * 100 if total_files > 0 else 0
        summary = f"\nSummary:\nTotal Files: {total_files}\nCorrect: {correct_predictions}\nAccuracy: {accuracy:.2f}%\n"
        f.write(summary)
        print(summary)
        print(f"Log saved to {log_filename}")

if __name__ == "__main__":
    main()
