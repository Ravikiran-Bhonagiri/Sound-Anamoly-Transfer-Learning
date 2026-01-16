# TFLite Model Deployment - Debugging Guide

This guide helps you troubleshoot and fix common deployment issues when deploying TensorFlow Lite models for sound anomaly detection on edge devices (Raspberry Pi, Docker containers, etc.).

---

## 🚨 Quick Diagnosis

**Got an error during inference?** Start here:

### Step 1: Identify Your Error Type

| Error Message | Issue Type | Jump To |
|---------------|-----------|---------|
| `ValueError: Dimension mismatch` | Model Export Problem | [Section 1](#1-dimension-mismatch-errors) |
| `ModuleNotFoundError: tensorflow` | Missing Dependencies | [Section 2](#2-missing-dependencies) |
| `Model not found` | Path/File Issues | [Section 3](#3-model-not-found-errors) |
| `PortAudioError` | Audio Device Issues | [Section 4](#4-audio-device-errors) |
| Inference too slow | Performance Issues | [Section 5](#5-performance-issues) |

### Step 2: Run the Validation Script

```bash
# Quick validation (recommended first step)
python scripts/validate_deployment.py
```

This will automatically check:
- ✅ All dependencies are installed
- ✅ Model file exists and loads correctly
- ✅ Model has correct input/output shapes
- ✅ Inference works end-to-end

---

## 1. Dimension Mismatch Errors

### 🔴 Error Example
```
ValueError: Cannot set tensor: Dimension mismatch. 
Got 16000 but expected 1 for dimension 0 of input
```

### 🎯 Root Cause
Your TFLite model was **exported incorrectly**. The model expects input shape `[1]` but should expect `[1, 16000]` or `[1, 16000, 1]` for audio data.

### ✅ Solution

#### Option A: Check Your Current Model (Fastest)
```bash
python scripts/validate_deployment.py
```

Look for the output:
```
📥 INPUT TENSOR:
  Shape:  [1 16000]  ← Good! This is correct
  # OR
  Shape:  [1]        ← Bad! This is wrong
```

#### Option B: Re-export the Model (Fix)

1. **Find your trained Keras model**:
   ```bash
   # Usually in artifacts directory
   ls artifacts/*.h5
   # OR
   ls artifacts/*/*.h5
   ```

2. **Re-convert to TFLite correctly**:
   ```python
   import tensorflow as tf
   import numpy as np
   
   # Load Keras model
   model = tf.keras.models.load_model('path/to/model.h5')
   
   # VERIFY input shape before conversion
   print(f"Keras model input shape: {model.input_shape}")
   # Should show: (None, 16000, 1) or similar
   
   # Convert to TFLite
   converter = tf.lite.TFLiteConverter.from_keras_model(model)
   tflite_model = converter.convert()
   
   # Save
   with open('sound_classifier.tflite', 'wb') as f:
       f.write(tflite_model)
   
   # VALIDATE the conversion
   interpreter = tf.lite.Interpreter(model_path='sound_classifier.tflite')
   interpreter.allocate_tensors()
   input_details = interpreter.get_input_details()
   
   print(f"TFLite input shape: {input_details[0]['shape']}")
   # Should still show: [1, 16000, 1]
   ```

3. **Verify the fix**:
   ```bash
   python scripts/validate_deployment.py
   ```

### 🔍 Prevention
Always validate your TFLite model immediately after conversion:
```python
# After conversion, ALWAYS check:
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
print("Input shape:", interpreter.get_input_details()[0]['shape'])
```

---

## 2. Missing Dependencies

### 🔴 Error Example
```
ModuleNotFoundError: No module named 'tensorflow'
# OR
ModuleNotFoundError: No module named 'tflite_runtime'
```

### 🎯 Root Cause
TensorFlow or TFLite Runtime is not installed in your Python environment.

### ✅ Solution

#### For Development/Training (Full TensorFlow)
```bash
pip install tensorflow>=2.13.0
pip install numpy>=1.21.0
```

#### For Inference Only (Lightweight - Recommended for Raspberry Pi)
```bash
pip install tflite-runtime
pip install numpy>=1.21.0
```

#### For Audio Processing
```bash
# Option 1: sounddevice (recommended)
pip install sounddevice

# Option 2: pyaudio
pip install pyaudio
```

### 📝 Create Requirements File

**For Training Environment** (`requirements_training.txt`):
```
tensorflow>=2.13.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
```

**For Inference Environment** (`requirements_inference.txt`):
```
tflite-runtime
numpy>=1.21.0
sounddevice>=0.4.6
```

Install with:
```bash
pip install -r requirements_inference.txt
```

### 🔍 Verify Installation
```bash
# Check what's installed
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
# OR
python -c "import tflite_runtime; print('TFLite Runtime:', tflite_runtime.__version__)"

# Check NumPy
python -c "import numpy; print('NumPy:', numpy.__version__)"
```

---

## 3. Model Not Found Errors

### 🔴 Error Example
```
FileNotFoundError: [Errno 2] No such file or directory: '/app/test/artifacts/sound_classifier.tflite'
```

### 🎯 Root Cause
The model file doesn't exist at the specified path, or the path is incorrect.

### ✅ Solution

#### Step 1: Locate Your Model
```bash
# Search for .tflite files in project
find . -name "*.tflite"

# On Windows PowerShell
Get-ChildItem -Recurse -Filter "*.tflite"

# Check artifacts directory
ls -lh artifacts/
ls -lh artifacts/*/  # Check timestamped subdirectories
```

#### Step 2: Use Absolute Paths
```python
# ❌ BAD: Relative path might break
model_path = "./artifacts/model.tflite"

# ✅ GOOD: Absolute path
import os
from pathlib import Path

# Method 1: Using pathlib
model_path = Path(__file__).parent / "artifacts" / "sound_classifier.tflite"

# Method 2: Using os
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "artifacts", "sound_classifier.tflite")

# Verify it exists
assert os.path.exists(model_path), f"Model not found: {model_path}"
```

#### Step 3: Docker Volume Mounts
If running in Docker, ensure volumes are mounted correctly:

```bash
# ❌ BAD: Model not mounted
docker run sound-anomaly-inference

# ✅ GOOD: Mount model as volume
docker run -v "/host/path/to/model.tflite:/app/model.tflite" \
    sound-anomaly-inference --model /app/model.tflite
```

### 🔍 Debug Path Issues
Add this to your inference script:
```python
import os

print("Current working directory:", os.getcwd())
print("Script location:", os.path.abspath(__file__))
print("Model path:", model_path)
print("Model exists?", os.path.exists(model_path))

if not os.path.exists(model_path):
    # List what's actually in the directory
    parent_dir = os.path.dirname(model_path)
    print(f"Contents of {parent_dir}:")
    print(os.listdir(parent_dir))
```

---

## 4. Audio Device Errors

### 🔴 Error Example
```
PortAudioError: Error opening audio device
# OR
OSError: [Errno -9996] Invalid input device
```

### 🎯 Root Cause
- No microphone connected
- Wrong audio device index
- Missing audio system libraries (ALSA on Linux)
- Docker container doesn't have access to audio devices

### ✅ Solution

#### Step 1: List Available Audio Devices
```python
import sounddevice as sd

# List all audio devices
print(sd.query_devices())

# Expected output:
#   0 Built-in Microphone, ALSA (2 in, 0 out)
#   1 Built-in Speaker, ALSA (0 in, 2 out)
```

#### Step 2: Select Correct Device
```python
# Get default input device
default_device = sd.default.device[0]
print(f"Using device {default_device}")

# OR specify device explicitly
device_index = 0  # Change to your microphone index
sd.default.device = device_index
```

#### Step 3: Docker Setup (Linux)
```bash
# Grant Docker access to audio devices
docker run --device /dev/snd \
    -v /path/to/model.tflite:/app/model.tflite \
    sound-anomaly-inference
```

#### Step 4: Install System Dependencies

**Ubuntu/Debian**:
```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-pyaudio
sudo apt-get install alsa-utils  # For ALSA support
```

**Raspberry Pi**:
```bash
sudo apt-get install portaudio19-dev
pip3 install sounddevice
```

### 🔍 Test Audio Capture
```python
import sounddevice as sd
import numpy as np

# Test recording 1 second of audio
duration = 1  # seconds
sample_rate = 16000

try:
    print(f"Recording {duration}s at {sample_rate}Hz...")
    audio = sd.rec(int(duration * sample_rate), 
                   samplerate=sample_rate, 
                   channels=1, 
                   dtype='float32')
    sd.wait()
    print(f"✅ Captured {len(audio)} samples")
    print(f"   Range: [{audio.min():.3f}, {audio.max():.3f}]")
except Exception as e:
    print(f"❌ Audio capture failed: {e}")
```

---

## 5. Performance Issues

### 🔴 Problem
Inference is too slow for real-time processing.

### 🎯 Root Cause
- Model is too large
- CPU-only execution on resource-constrained device
- Inefficient preprocessing

### ✅ Solution

#### Step 1: Measure Current Performance
```python
import time
import numpy as np

# Simulate inference timing
num_iterations = 100
times = []

for _ in range(num_iterations):
    start = time.perf_counter()
    
    # Your inference code here
    interpreter.set_tensor(input_details[0]['index'], audio_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    end = time.perf_counter()
    times.append((end - start) * 1000)  # Convert to ms

mean_time = np.mean(times)
print(f"Average inference time: {mean_time:.2f} ms")
print(f"Throughput: {1000/mean_time:.2f} inferences/sec")

# For 1 second of audio @ 16kHz
audio_duration_ms = 1000  # 1 second
if mean_time < audio_duration_ms:
    print(f"✅ Real-time capable ({audio_duration_ms/mean_time:.1f}x faster)")
else:
    print(f"❌ Not real-time ({mean_time/audio_duration_ms:.1f}x slower)")
```

#### Step 2: Enable Quantization
Convert model to int8 for 4x speedup:

```python
import tensorflow as tf

# Load Keras model
model = tf.keras.models.load_model('model.h5')

# Create converter with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Provide representative dataset
def representative_dataset():
    for _ in range(100):
        # Generate sample audio data
        data = np.random.randn(1, 16000, 1).astype(np.float32)
        yield [data]

converter.representative_dataset = representative_dataset

# Convert
tflite_quant_model = converter.convert()

# Save
with open('model_quantized.tflite', 'wb') as f:
    f.write(tflite_quant_model)
```

#### Step 3: Optimize Preprocessing
```python
# ❌ SLOW: Multiple operations
audio = normalize(filter(resample(raw_audio)))

# ✅ FAST: Minimal processing
audio = raw_audio / 32768.0  # Simple normalization
audio = audio[:16000]  # Simple truncation
```

#### Step 4: Use Threading (Advanced)
```python
import queue
import threading

# Audio capture in separate thread
audio_queue = queue.Queue(maxsize=5)

def audio_capture_thread():
    while True:
        audio = sd.rec(16000, samplerate=16000, channels=1)
        sd.wait()
        audio_queue.put(audio)

# Inference in main thread
def inference_thread():
    while True:
        audio = audio_queue.get()
        # Run inference
        predictions = run_inference(audio)
        handle_results(predictions)
```

---

## 6. Preprocessing Mismatches

### 🔴 Problem
Model works in training but gives wrong predictions in production.

### 🎯 Root Cause
Preprocessing during inference doesn't match training preprocessing.

### ✅ Solution

#### Step 1: Document Training Preprocessing
Create a `PREPROCESSING.md` file:
```markdown
# Model Preprocessing

## Audio Parameters
- Sample Rate: 16000 Hz
- Duration: 1.0 seconds
- Channels: 1 (mono)
- Samples: 16000

## Normalization
- Method: Min-Max to [-1, 1]
- Formula: audio / max(abs(audio))

## Shape
- Input: [1, 16000, 1]
- Type: float32
```

#### Step 2: Create Reusable Preprocessing Function
```python
def preprocess_audio(raw_audio, sample_rate=16000, duration=1.0):
    """
    Preprocess audio for model inference.
    MUST match training preprocessing exactly!
    
    Args:
        raw_audio: numpy array of audio samples
        sample_rate: Expected sample rate (Hz)
        duration: Expected duration (seconds)
    
    Returns:
        Preprocessed audio ready for model input
    """
    target_length = int(sample_rate * duration)
    
    # 1. Ensure correct length
    if len(raw_audio) > target_length:
        audio = raw_audio[:target_length]
    else:
        # Pad with zeros if too short
        audio = np.pad(raw_audio, (0, target_length - len(raw_audio)))
    
    # 2. Normalize (MATCH YOUR TRAINING!)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    
    # 3. Reshape to model input shape
    audio = audio.reshape(1, target_length, 1).astype(np.float32)
    
    return audio
```

#### Step 3: Validate Preprocessing
```python
# Test your preprocessing
test_audio = np.random.randn(16000)
processed = preprocess_audio(test_audio)

print(f"Shape: {processed.shape}")  # Should be [1, 16000, 1]
print(f"Type: {processed.dtype}")   # Should be float32
print(f"Range: [{processed.min():.3f}, {processed.max():.3f}]")
```

---

## 🛠️ Validation Checklist

Before deploying to production, verify:

- [ ] **Dependencies**: All required packages installed
  ```bash
  python scripts/validate_deployment.py
  ```

- [ ] **Model File**: Exists and loads without errors
  ```bash
  ls -lh artifacts/sound_classifier.tflite
  ```

- [ ] **Input Shape**: Correct for audio data (≥8000 samples)
  ```python
  # Should show [1, 16000] or [1, 16000, 1]
  print(input_details[0]['shape'])
  ```

- [ ] **Test Inference**: Works with synthetic data
  ```bash
  python scripts/validate_deployment.py
  ```

- [ ] **Audio Capture**: Microphone accessible
  ```python
  import sounddevice as sd
  sd.query_devices()
  ```

- [ ] **Performance**: Real-time capable
  ```
  Inference time < Audio duration
  ```

- [ ] **Preprocessing**: Matches training exactly
  ```python
  # Document and test preprocessing function
  ```

---

## 📚 Additional Resources

### Scripts Available

| Script | Purpose | Command |
|--------|---------|---------|
| `validate_deployment.py` | Complete validation | `python scripts/validate_deployment.py` |
| `check_dependencies.sh` | Check system deps | `bash scripts/check_dependencies.sh` |
| `quick_diagnosis.sh` | Quick model check | `bash scripts/quick_diagnosis.sh` |

### Useful Commands

```bash
# Check Python environment
python --version
pip list | grep -i tensor

# Find model files
find . -name "*.tflite"
find . -name "*.h5"

# Test audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Check disk space
df -h

# Monitor system resources
top    # Linux
htop   # Linux (better)
```

### Getting Help

1. **Run validation script**: `python scripts/validate_deployment.py`
2. **Check this guide**: Find your error in Section 1-6
3. **Review logs**: Look for stack traces and error messages
4. **Verify prerequisites**: Ensure all dependencies installed

---

## 🎯 Common Deployment Patterns

### Pattern 1: Local Development
```bash
# 1. Train model
python train.py

# 2. Validate model
python scripts/validate_deployment.py

# 3. Test inference
python inference.py --model artifacts/sound_classifier.tflite
```

### Pattern 2: Docker Deployment
```bash
# 1. Build image
docker build -f Dockerfile.inference -t sound-anomaly-inference .

# 2. Run with model mounted
docker run --device /dev/snd \
    -v "${PWD}/artifacts/sound_classifier.tflite:/app/model.tflite" \
    sound-anomaly-inference --model /app/model.tflite
```

### Pattern 3: Raspberry Pi Deployment
```bash
# 1. Transfer model
scp artifacts/sound_classifier.tflite pi@raspberrypi:/home/pi/

# 2. Install dependencies
ssh pi@raspberrypi "pip3 install tflite-runtime numpy sounddevice"

# 3. Transfer inference script
scp inference.py pi@raspberrypi:/home/pi/

# 4. Run
ssh pi@raspberrypi "python3 inference.py --model /home/pi/sound_classifier.tflite"
```

---

**Need more help?** Run `python scripts/validate_deployment.py` for automated diagnostics.
