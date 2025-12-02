# Industrial Audio Anomaly Detection

This project implements an end-to-end audio classification system designed for the Raspberry Pi. It uses a fine-tuned YAMNet model to detect machine states from audio input.

**Repository:** [Sound-Anamoly-Transfer-Learning](https://github.com/Ravikiran-Bhonagiri/Sound-Anamoly-Transfer-Learning.git)

## Project Structure

The repository is organized as follows:

```
.
├── artifacts/                  # Generated models and plots
│   ├── sound_classifier.tflite
│   ├── sound_classifier_head.h5
│   ├── embeddings_pca_check.png
│   └── confusion_matrix.png
├── data/                       # Dataset directory (ignored in git)
├── venv/                       # Python virtual environment (ignored in git)
├── Dockerfile                  # Configuration for building the Docker image
├── README.md                   # This documentation file
├── train.py                    # Script to train the model
├── inference.py                # Script for real-time inference on Raspberry Pi
├── visualize_data.py           # Script to visualize data quality (PCA/t-SNE)
├── evaluate_model.py           # Script to calculate model metrics (Accuracy, F1, etc.)
└── test_docker_inference.py    # Script to verify the Docker image functionality
```

## Detailed File Descriptions

### 1. Core Scripts

*   **`train.py`**:
    *   **Purpose**: Trains the audio classification model.
    *   **Process**: Loads audio from `data/`, splits it into 1-second chunks, extracts embeddings using YAMNet (TFHub), trains a custom classifier head, and exports the full model to `sound_classifier.tflite`.
    *   **Output**: `artifacts/sound_classifier.tflite`, `artifacts/sound_classifier_head.h5`.

*   **`inference.py`**:
    *   **Purpose**: Runs real-time inference on the Raspberry Pi.
    *   **Features**:
        *   **Multithreading**: Uses a thread-safe `queue` to decouple audio capture from inference, ensuring no data loss.
        *   **Real-time**: Processes audio in 1-second chunks with minimal latency.
        *   **Instrumentation**: Prints inference time (ms) for performance monitoring.
    *   **Process**: Captures audio from the microphone using `sounddevice`, downsamples it to 16kHz, maintains a rolling 1-second buffer, and runs inference using the TFLite model.
    *   **Usage**: `python inference.py` (usually run inside Docker).

*   **`visualize_data.py`**:
    *   **Purpose**: Analyzes the quality of the dataset.
    *   **Process**: Extracts embeddings and uses PCA (Principal Component Analysis) to reduce dimensionality to 2D for plotting. Calculates the Silhouette Score to measure class separability.
    *   **Output**: `artifacts/embeddings_pca_check.png`.

### 2. Evaluation & Testing

*   **`evaluate_model.py`**:
    *   **Purpose**: Evaluates the trained model's performance.
    *   **Process**: Runs the model on the entire dataset and calculates Accuracy, Precision, Recall, and F1-Score. Generates a confusion matrix.
    *   **Output**: Console report and `artifacts/confusion_matrix.png`.

*   **`test_docker_inference.py`**:
    *   **Purpose**: Verifies that the Docker image is working correctly.
    *   **Process**: Runs inside the container, loads a WAV file, and performs inference using the installed dependencies (without `scipy` to save space).
    *   **Performance**: Measures and reports inference time per chunk.

### 3. Deployment

*   **`Dockerfile`**:
    *   **Purpose**: Defines the environment for the Raspberry Pi.
    *   **Features**:
        *   **Multi-stage Build**: Uses a builder stage to minimize final image size (~337MB).
        *   **Optimized Dependencies**: Installs `tflite-runtime`, `numpy`, and `sounddevice`.

## Usage Guide

### Prerequisites
-   Python 3.9+
-   Docker (for deployment)

### Training the Model
```bash
python train.py
```

### Evaluating the Model
```bash
python evaluate_model.py
```

### Building the Docker Image
```bash
docker build -t sound_classifier .
```

### Testing the Docker Image
To verify the image works by running the test script against a sample file:

**Windows (PowerShell):**
```powershell
docker run --rm -v "${PWD}:/app/test" sound_classifier python /app/test/test_docker_inference.py /app/test/data/on_state/on_merged.wav
```

**Linux/Mac:**
```bash
docker run --rm -v "$(pwd):/app/test" sound_classifier python /app/test/test_docker_inference.py /app/test/data/on_state/on_merged.wav
```

**Expected Output:**
```
Time 400.00s: on_state (1.00) | Chunk: (16000,) | Inference: 12.35ms
```
*Inference time is typically ~12-15ms per 1-second chunk.*

### Running on Raspberry Pi
```bash
# Run with audio device access
docker run -it --rm --device /dev/snd sound_classifier
```

## Model Details
-   **Base Model**: YAMNet (Transfer Learning)
-   **Input**: 1-second audio clips at 16kHz
-   **Classes**: `on_state`, `off_state`, `solid_state`, `soft_state`
-   **Format**: TensorFlow Lite (`.tflite`)

## Appendix: Dockerfile Configuration

For reference, here is the optimized multi-stage `Dockerfile` used in this project:

```dockerfile
# Stage 1: Builder
FROM python:3.9-slim as builder

WORKDIR /app

# Install dependencies to a specific location
# We use --target to install into a specific directory that we can copy later
RUN pip install --no-cache-dir --target=/install \
    "numpy<2" \
    sounddevice \
    tflite-runtime

# Stage 2: Runtime
FROM python:3.9-slim

# Install runtime system dependencies
# libportaudio2 is required for sounddevice
RUN apt-get update && apt-get install -y \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed python packages from builder
# We add the install directory to PYTHONPATH so Python can find them
COPY --from=builder /install /usr/local/lib/python3.9/site-packages

# Copy inference script
COPY inference.py .

# Create artifacts directory and copy model
RUN mkdir artifacts
COPY artifacts/sound_classifier.tflite artifacts/

# Command to run
CMD ["python", "inference.py"]
```
