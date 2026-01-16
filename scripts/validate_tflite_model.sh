#!/bin/bash
# File: validate_tflite_model.sh
# Purpose: Comprehensive TFLite model validation

# Auto-detect paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Validating TFLite Model..."

# Use env var, arg, or default
TFLITE_MODEL_PATH="${TFLITE_MODEL_PATH:-${1:-$PROJECT_ROOT/artifacts/sound_classifier.tflite}}"

python3 << EOF
import sys
import os
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# Configuration
TFLITE_MODEL_PATH = "$TFLITE_MODEL_PATH"

print("="*70)
print("TFLITE MODEL VALIDATION")
print("="*70)
print(f"Model path: {TFLITE_MODEL_PATH}\n")

if not os.path.exists(TFLITE_MODEL_PATH):
    print(f"❌ Model not found at {TFLITE_MODEL_PATH}")
    sys.exit(1)

# Load model
try:
    interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    print("✅ Model loaded successfully\n")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print details
print("📥 INPUT TENSOR DETAILS:")
print("-" * 70)
for i, detail in enumerate(input_details):
    print(f"  Input[{i}]:")
    print(f"    Name:           {detail['name']}")
    print(f"    Shape:          {detail['shape']}")
    print(f"    Type:           {detail['dtype']}")
    print(f"    Index:          {detail['index']}")
    print(f"    Quantization:   {detail.get('quantization', 'None')}")

print("\n📤 OUTPUT TENSOR DETAILS:")
print("-" * 70)
for i, detail in enumerate(output_details):
    print(f"  Output[{i}]:")
    print(f"    Name:           {detail['name']}")
    print(f"    Shape:          {detail['shape']}")
    print(f"    Type:           {detail['dtype']}")
    print(f"    Index:          {detail['index']}")
    print(f"    Quantization:   {detail.get('quantization', 'None')}")

# Validation checks
print("\n🔍 VALIDATION CHECKS:")
print("-" * 70)

input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']
output_shape = output_details[0]['shape']

checks_passed = 0
total_checks = 5

# Check 1: Input shape validity for audio
print(f"\n[Check 1/{total_checks}] Input Shape for Audio Data")
if len(input_shape) >= 2:
    audio_length = input_shape[1] if len(input_shape) >= 2 else None
    if audio_length and audio_length >= 8000:
        print(f"  ✅ PASS: Input shape {input_shape} is suitable for audio")
        print(f"     Audio samples: {audio_length}")
        checks_passed += 1
    else:
        print(f"  ❌ FAIL: Input shape {input_shape} seems too small for audio")
        print(f"     Expected at least 8000 samples, got {audio_length}")
else:
    print(f"  ❌ FAIL: Input shape {input_shape} is invalid for audio")
    print(f"     Expected shape like [1, 16000] or [1, 16000, 1]")

# Check 2: Input dtype
print(f"\n[Check 2/{total_checks}] Input Data Type")
if input_dtype == np.float32:
    print(f"  ✅ PASS: Input dtype is float32 (standard)")
    checks_passed += 1
elif input_dtype == np.int8 or input_dtype == np.uint8:
    print(f"  ⚠️  WARN: Input dtype is {input_dtype} (quantized)")
    print(f"     Ensure preprocessing includes quantization")
    checks_passed += 1
else:
    print(f"  ❌ FAIL: Unexpected input dtype: {input_dtype}")

# Check 3: Output shape for classification
print(f"\n[Check 3/{total_checks}] Output Shape for Classification")
if len(output_shape) == 2:
    num_classes = output_shape[1]
    print(f"  ✅ PASS: Output shape {output_shape} is valid")
    print(f"     Number of classes: {num_classes}")
    checks_passed += 1
else:
    print(f"  ⚠️  WARN: Output shape {output_shape} is unusual")
    print(f"     Expected shape like [1, num_classes]")

# Check 4: Test inference
print(f"\n[Check 4/{total_checks}] Test Inference")
try:
    test_input = np.random.randn(*input_shape).astype(input_dtype)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"  ✅ PASS: Inference successful")
    print(f"     Input shape: {test_input.shape}")
    print(f"     Output shape: {output.shape}")
    print(f"     Sample output: {output[0][:5] if len(output[0]) > 5 else output[0]}")
    checks_passed += 1
except Exception as e:
    print(f"  ❌ FAIL: Inference failed")
    print(f"     Error: {e}")

# Check 5: Model file size
print(f"\n[Check 5/{total_checks}] Model File Size")
file_size = os.path.getsize(TFLITE_MODEL_PATH)
file_size_kb = file_size / 1024
file_size_mb = file_size_kb / 1024

if file_size_kb > 10:
    print(f"  ✅ PASS: Model size is reasonable")
    print(f"     Size: {file_size_mb:.2f} MB ({file_size_kb:.2f} KB)")
    checks_passed += 1
else:
    print(f"  ⚠️  WARN: Model seems very small")
    print(f"     Size: {file_size_kb:.2f} KB")

# Summary
print("\n" + "="*70)
print(f"VALIDATION SUMMARY: {checks_passed}/{total_checks} checks passed")
print("="*70)

if checks_passed == total_checks:
    print("✅ ALL CHECKS PASSED - Model is ready for deployment!")
    sys.exit(0)
elif checks_passed >= total_checks * 0.8:
    print("⚠️  MOST CHECKS PASSED - Review warnings before deployment")
    sys.exit(0)
else:
    print("❌ VALIDATION FAILED - Fix issues before deployment")
    sys.exit(1)
EOF
