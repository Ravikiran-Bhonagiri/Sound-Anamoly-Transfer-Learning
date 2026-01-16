#!/bin/bash
# File: quick_diagnosis.sh
# Purpose: Quickly diagnose TFLite model issues

# Auto-detect script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Sound Anomaly Model - Quick Diagnosis"
echo "=========================================="

# Check Python environment
echo -e "\n[1/5] Checking Python environment..."
python3 --version 2>/dev/null || python --version || echo "❌ Python not found"

# Check TFLite installation
echo -e "\n[2/5] Checking TFLite Runtime..."
python3 -c "import tflite_runtime.interpreter as tflite; print('✅ tflite-runtime installed')" 2>/dev/null || \
python3 -c "import tensorflow.lite as tflite; print('✅ tensorflow installed')" 2>/dev/null || \
echo "❌ Neither tflite-runtime nor tensorflow found"

# Check model file - use env var or default to project artifacts
echo -e "\n[3/5] Checking model file..."
MODEL_PATH="${TFLITE_MODEL_PATH:-${1:-$PROJECT_ROOT/artifacts/sound_classifier.tflite}}"
if [ -f "$MODEL_PATH" ]; then
    echo "✅ Model found at: $MODEL_PATH"
    ls -lh "$MODEL_PATH"
else
    echo "❌ Model not found at: $MODEL_PATH"
    echo "   Set TFLITE_MODEL_PATH or provide path as argument"
    exit 1
fi

# Validate model structure
echo -e "\n[4/5] Validating TFLite model structure..."
python3 << EOF
import sys
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

model_path = "$MODEL_PATH"

try:
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"✅ Model loaded successfully")
    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Input dtype: {input_details[0]['dtype']}")
    print(f"   Output shape: {output_details[0]['shape']}")
    
    # Check for common issues
    input_shape = input_details[0]['shape']
    if len(input_shape) == 1 or (len(input_shape) == 2 and input_shape[1] == 1):
        print("\n⚠️  WARNING: Input shape looks incorrect!")
        print(f"   Current shape: {input_shape}")
        print(f"   Expected shape for audio: [1, 16000] or [1, 16000, 1]")
        print("\n   ACTION REQUIRED: Re-export the Keras model to TFLite")
        sys.exit(1)
    else:
        print("✅ Input shape looks valid for audio data")
        
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)
EOF

echo -e "\n[5/5] Checking audio dependencies..."
python3 -c "import sounddevice; print('✅ sounddevice installed')" 2>/dev/null || \
python3 -c "import pyaudio; print('✅ pyaudio installed')" 2>/dev/null || \
echo "⚠️  No audio library found (sounddevice or pyaudio needed)"

echo -e "\n=========================================="
echo "Diagnosis complete!"
echo "=========================================="
