# Sound Anomaly Detection - Testing & Usage Guide

This document provides a comprehensive guide to building, running, and verifying the Sound Anomaly Detection system using Docker.

## 1. System Overview

The system consists of two optimized Docker images:

| Image Name | Purpose | Size | Key Features |
| :--- | :--- | :--- | :--- |
| **`sound-anomaly-training`** | Model Training | ~1.79 GB | Includes TensorFlow (CPU). Independent of data (data is mounted). Generates timestamped artifacts. |
| **`sound-anomaly-inference`** | Real-time Inference | ~311 MB | Lightweight (TFLite Runtime). Generic (Model is mounted). Optimized for Raspberry Pi. |

---

## 2. Prerequisites

1.  **Docker**: Ensure Docker is installed and running on your machine.
2.  **Data Directory**: You must have a folder containing your training audio files.
    *   Structure:
        ```text
        data/
        ├── on_state/
        │   ├── audio1.wav
        │   └── ...
        ├── off_state/
        └── ...
        ```

---

## 3. Build Instructions

You must build the Docker images before running any scripts.

### 3.1 Build Training Image
```bash
docker build -f Dockerfile.training -t sound-anomaly-training .
```

### 3.2 Build Inference Image
```bash
docker build -f Dockerfile.inference -t sound-anomaly-inference .
```

---

## 4. Automated Workflow (Recommended)

We provide automation scripts to handle the full lifecycle: **Train $\rightarrow$ Locate Model $\rightarrow$ Verify**.

### 🐧 Linux / Raspberry Pi (`run_pipeline.sh`)

**Setup**:
```bash
chmod +x run_pipeline.sh
```

**Usage**:
```bash
./run_pipeline.sh [OPTIONS]
```

**Arguments**:
| Flag | Long Flag | Description | Default |
| :--- | :--- | :--- | :--- |
| `-d` | `--data` | Path to training data directory | `./data` |
| `-t` | `--test` | Path to `.wav` file for verification | `./data/on_state/on_merged.wav` |
| `-a` | `--artifacts` | Directory to save model outputs | `./artifacts` |

**Example**:
```bash
./run_pipeline.sh --data /home/pi/my_data --test /home/pi/test_audio.wav
```

### 🪟 Windows PowerShell (`run_pipeline.ps1`)

**Usage**:
```powershell
.\run_pipeline.ps1 -DataDir "C:\path\to\data" -TestFile "C:\path\to\test.wav"
```

**Arguments**:
| Parameter | Description | Default |
| :--- | :--- | :--- |
| `-DataDir` | Path to training data directory | `.\data` |
| `-TestFile` | Path to `.wav` file for verification | `.\data\on_state\on_merged.wav` |
| `-ArtifactsDir` | Directory to save model outputs | `.\artifacts` |

---

## 5. Manual Workflow (Advanced)

If you need granular control, you can run the Docker containers manually.

### Step 1: Run Training
Mount your local data and artifacts folders. The container will create a new timestamped folder (e.g., `artifacts/2025_12_04_14_30_00/`).

```bash
docker run --rm \
    -v "/abs/path/to/data:/app/data" \
    -v "/abs/path/to/artifacts:/app/artifacts" \
    sound-anomaly-training
```

### Step 2: Run Inference
To use the model you just trained, you must mount it into the inference container.

**Command Syntax**:
```bash
docker run --rm \
    --device /dev/snd \
    -v "/abs/path/to/test_data:/app/test_data" \
    -v "/abs/path/to/model.tflite:/app/model.tflite" \
    sound-anomaly-inference \
    --model /app/model.tflite
```

*   **Note**: `--device /dev/snd` is only required if you are using a microphone. For file-based testing, it can be omitted.

---

## 6. Troubleshooting

| Issue | Possible Cause | Solution |
| :--- | :--- | :--- |
| `image not found` | Docker image not built | Run the **Build Instructions** commands first. |
| `volume mount failed` | Path does not exist | Ensure you are using **absolute paths** or correct relative paths. |
| `permission denied` | Linux file permissions | Run with `sudo` or check folder ownership (`chown`). |
| `PortAudioError` | No microphone detected | Ensure `--device /dev/snd` is passed (Linux only) and a mic is connected. |
