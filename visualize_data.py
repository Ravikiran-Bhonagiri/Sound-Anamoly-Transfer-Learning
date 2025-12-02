import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import wavfile
from scipy import signal

# Configuration
DATA_DIR = 'data'
CLASSES = ['on_state', 'off_state', 'solid_state', 'soft_state']
TARGET_SR = 16000
YAMNET_MODEL_HANDLE = 'https://tfhub.dev/google/yamnet/1'

def load_yamnet():
    """Loads the YAMNet model from TFHub."""
    print("Loading YAMNet model...")
    model = hub.load(YAMNET_MODEL_HANDLE)
    return model

def load_and_preprocess_wav(file_path):
    """Loads a WAV file, resamples to 16kHz, and normalizes."""
    # Load using scipy
    sr, wav_data = wavfile.read(file_path)
    
    # Normalize to [-1, 1] if not float
    if wav_data.dtype != np.float32:
        wav_data = wav_data / np.iinfo(wav_data.dtype).max
        wav_data = wav_data.astype(np.float32)

    # Resample if necessary
    if sr != TARGET_SR:
        number_of_samples = round(len(wav_data) * float(TARGET_SR) / sr)
        wav_data = signal.resample(wav_data, number_of_samples)
    
    return wav_data

def get_embeddings(model, file_paths):
    """Extracts embeddings for 1-second chunks from a list of files."""
    embeddings = []
    labels = []
    chunk_size = 16000  # 1 second at 16kHz
    
    for file_path, label in file_paths:
        try:
            wav_data = load_and_preprocess_wav(file_path)
            
            # Split into 1-second chunks
            for start in range(0, len(wav_data) - chunk_size + 1, chunk_size):
                chunk = wav_data[start:start+chunk_size]
                
                # YAMNet expects a 1D tensor
                scores, embeddings_spectrogram, spectrogram = model(chunk)
                
                # embeddings_spectrogram shape: (N, 1024) where N is number of frames
                # For 1 second, we might get 1 or 2 frames. Average them.
                mean_embedding = tf.reduce_mean(embeddings_spectrogram, axis=0)
                
                embeddings.append(mean_embedding.numpy())
                labels.append(label)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    return np.array(embeddings), np.array(labels)

from sklearn.metrics import silhouette_score

def visualize_embeddings(embeddings, labels):
    """Reduces dimensionality using PCA and plots the results."""
    
    # Calculate Silhouette Score on the high-dimensional embeddings
    sil_score = silhouette_score(embeddings, labels)
    print(f"Silhouette Score: {sil_score:.4f}")
    
    if sil_score > 0.5:
        print("Interpretation: Excellent separation. Classes are very distinct.")
    elif sil_score > 0.2:
        print("Interpretation: Good separation. Some overlap may exist but classes are distinguishable.")
    else:
        print("Interpretation: Poor separation. Classes are overlapping significantly.")

    print("Running PCA...")
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Calculate explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained Variance Ratio: {explained_variance}")
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=embeddings_2d[:, 0], 
        y=embeddings_2d[:, 1], 
        hue=labels, 
        palette='viridis',
        s=100,
        alpha=0.8
    )
    
    plt.title(f'YAMNet Embeddings PCA (Silhouette Score: {sil_score:.2f})')
    plt.xlabel(f'PC1 ({explained_variance[0]:.2%} Variance)')
    plt.ylabel(f'PC2 ({explained_variance[1]:.2%} Variance)')
    plt.legend(title='Classes')
    plt.grid(True, alpha=0.3)
    
    output_path = 'artifacts/embeddings_pca_check.png'
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")

def main():
    # Gather file paths
    file_paths = []
    for class_name in CLASSES:
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} does not exist.")
            continue
            
        for filename in os.listdir(class_dir):
            if filename.lower().endswith('.wav'):
                file_paths.append((os.path.join(class_dir, filename), class_name))
    
    if not file_paths:
        print("No WAV files found in data directories.")
        return

    print(f"Found {len(file_paths)} audio files.")
    
    # Load model and get embeddings
    model = load_yamnet()
    embeddings, labels = get_embeddings(model, file_paths)
    
    if len(embeddings) == 0:
        print("No embeddings extracted.")
        return

    # Visualize
    visualize_embeddings(embeddings, labels)

if __name__ == "__main__":
    main()
