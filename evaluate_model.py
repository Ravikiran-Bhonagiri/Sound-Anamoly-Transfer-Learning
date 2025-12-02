import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from scipy.io import wavfile
from scipy import signal
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration
DATA_DIR = 'data'
CLASSES = ['on_state', 'off_state', 'solid_state', 'soft_state']
TARGET_SR = 16000
YAMNET_MODEL_HANDLE = 'https://tfhub.dev/google/yamnet/1'
INPUT_LENGTH = 16000

def load_yamnet():
    return hub.load(YAMNET_MODEL_HANDLE)

def prepare_dataset():
    """Loads data, splits into chunks, and extracts embeddings."""
    print("Preparing dataset...")
    X = []
    y = []
    
    yamnet = load_yamnet()

    for i, class_name in enumerate(CLASSES):
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            continue
            
        for filename in os.listdir(class_dir):
            if not filename.lower().endswith('.wav'):
                continue
                
            filepath = os.path.join(class_dir, filename)
            try:
                sr, wav_data = wavfile.read(filepath)
                
                if wav_data.dtype != np.float32:
                    wav_data = wav_data / np.iinfo(wav_data.dtype).max
                    wav_data = wav_data.astype(np.float32)
                    
                if sr != TARGET_SR:
                    num_samples = int(len(wav_data) * TARGET_SR / sr)
                    wav_data = signal.resample(wav_data, num_samples)
                
                chunk_size = INPUT_LENGTH
                for start in range(0, len(wav_data) - chunk_size + 1, chunk_size):
                    chunk = wav_data[start:start+chunk_size]
                    scores, embeddings, spectrogram = yamnet(chunk)
                    mean_embedding = tf.reduce_mean(embeddings, axis=0)
                    X.append(mean_embedding.numpy())
                    y.append(i)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                
    return np.array(X), np.array(y)

def main():
    # 1. Load Data
    X, y = prepare_dataset()
    print(f"Total samples: {len(X)}")
    
    # 2. Load Model
    # We load the Keras head model
    model_path = 'artifacts/sound_classifier_head.h5'
    if not os.path.exists(model_path):
        print("Model not found.")
        return
        
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    
    # 3. Evaluate
    # Since we can't reproduce the exact train/val split from training,
    # we will evaluate on the entire dataset to get an overall performance check.
    # Note: This includes training data, so accuracy might be slightly optimistic,
    # but it confirms if the model learned the data.
    
    print("Running predictions...")
    y_pred_probs = model.predict(X)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # 4. Metrics
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=CLASSES))
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('artifacts/confusion_matrix.png')
    print("Confusion matrix saved to artifacts/confusion_matrix.png")

if __name__ == "__main__":
    main()
