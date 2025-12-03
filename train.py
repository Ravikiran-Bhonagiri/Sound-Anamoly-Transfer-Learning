import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt

# Configuration
DATA_DIR = 'data'
CLASSES = ['on_state', 'off_state', 'solid_state', 'soft_state']
TARGET_SR = 16000
YAMNET_MODEL_HANDLE = 'https://tfhub.dev/google/yamnet/1'
EPOCHS = 100
BATCH_SIZE = 32
INPUT_LENGTH = 16000  # 1 second at 16kHz

# YAMNet outputs embeddings every 0.48s. 
# However, for the custom head, we want to classify the *entire* clip or chunks of it.
# Strategy: We will extract embeddings for the files and train a simple classifier on the *mean* embedding 
# or individual frame embeddings.
# Given the user wants to detect "Machine State", a mean embedding over a short window is robust.
# But for real-time inference, we usually feed 0.975s chunks. 
# Let's align training with inference: Split training data into 0.975s chunks.

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
            print(f"Warning: {class_dir} not found.")
            continue
            
        for filename in os.listdir(class_dir):
            if not filename.lower().endswith('.wav'):
                continue
                
            filepath = os.path.join(class_dir, filename)
            sr, wav_data = wavfile.read(filepath)
            
            # Normalize
            if wav_data.dtype != np.float32:
                wav_data = wav_data / np.iinfo(wav_data.dtype).max
                wav_data = wav_data.astype(np.float32)
                
            # Resample
            if sr != TARGET_SR:
                num_samples = int(len(wav_data) * TARGET_SR / sr)
                wav_data = signal.resample(wav_data, num_samples)
            
            # Split into chunks of 1.0s (16000 samples)
            # This matches the inference buffer size
            chunk_size = 16000
            for start in range(0, len(wav_data) - chunk_size + 1, chunk_size):
                chunk = wav_data[start:start+chunk_size]
                
                # Get embedding from YAMNet
                # YAMNet returns (N, 1024) embeddings. For 0.975s, it returns 2 frames usually.
                # We will take the MEAN of these frames to get a single 1024 vector for this chunk.
                scores, embeddings, spectrogram = yamnet(chunk)
                mean_embedding = tf.reduce_mean(embeddings, axis=0)
                
                X.append(mean_embedding.numpy())
                y.append(i) # Label index
                
    return np.array(X), np.array(y)

def create_model():
    """Creates the classifier head."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    # 1. Prepare Data
    X, y = prepare_dataset()
    
    if len(X) == 0:
        print("No data found. Exiting.")
        return
        
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    # Shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # 2. Train Model
    model = create_model()
    
    # Callback to save the best model based on validation loss
    checkpoint_path = 'artifacts/sound_classifier_head.h5'
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    
    history = model.fit(
        X, y, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        validation_split=0.2,
        callbacks=[checkpoint_cb]
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    plt.savefig('artifacts/training_history.png')
    print("Saved training history plot to artifacts/training_history.png")
    
    # 3. Load Best Keras Model
    # We reload the best model saved by the checkpoint to ensure TFLite conversion uses the optimized weights
    model = tf.keras.models.load_model(checkpoint_path)
    print(f"Loaded best model from {checkpoint_path}")
    
    # 4. Convert to TFLite
    # We need to combine YAMNet + Head for the final TFLite?
    # Actually, running YAMNet on Pi in Python is heavy if not TFLite.
    # The user asked for "The final model must be converted to TFLite".
    # YAMNet is available as TFLite from TFHub, but we trained a separate head.
    # OPTION A: Run YAMNet TFLite -> Get Output -> Run Head TFLite.
    # OPTION B: Create a single Keras model that wraps YAMNet (as a layer) + Head, then convert.
    # Option B is better for a single file deployment.
    
    print("Building full end-to-end model for TFLite conversion...")
    
    # Define a custom model that includes YAMNet
    class AudioClassifier(tf.keras.Model):
        def __init__(self, yamnet_model, head_model):
            super(AudioClassifier, self).__init__()
            self.yamnet = yamnet_model
            self.head = head_model
            
        @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
        def call(self, waveform):
            # YAMNet expects raw waveform
            scores, embeddings, spectrogram = self.yamnet(waveform)
            # Embeddings shape: (N, 1024)
            # We take mean across frames for the chunk
            mean_embedding = tf.reduce_mean(embeddings, axis=0)
            # Add batch dimension for the head: (1, 1024)
            mean_embedding = tf.expand_dims(mean_embedding, 0)
            return self.head(mean_embedding)
            
    yamnet = load_yamnet()
    full_model = AudioClassifier(yamnet, model)
    
    # Convert
    converter = tf.lite.TFLiteConverter.from_keras_model(full_model)
    # Enable ops for YAMNet (some might be custom or require flex, but usually standard)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # Enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # Enable TensorFlow ops.
    ]
    tflite_model = converter.convert()
    
    with open('artifacts/sound_classifier.tflite', 'wb') as f:
        f.write(tflite_model)
        
    print("Saved artifacts/sound_classifier.tflite")

if __name__ == "__main__":
    main()
