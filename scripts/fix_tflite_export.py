"""
Fix TFLite Export Script
Re-exports Keras model to TFLite with correct input shape
"""

import tensorflow as tf
import numpy as np
import sys
from pathlib import Path

# ========================================
# CONFIGURATION
# ========================================

# Update this path to your Keras model
KERAS_MODEL_PATH = "artifacts/sound_classifier_head.h5"  
TFLITE_OUTPUT_PATH = "artifacts/sound_classifier_fixed.tflite"

# ========================================
# LOAD KERAS MODEL
# ========================================

print("="*70)
print("FIXING TFLITE MODEL EXPORT")
print("="*70)

print(f"\n[1/4] Loading Keras model from: {KERAS_MODEL_PATH}")

try:
    model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    print(f"✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading Keras model: {e}")
    print(f"\nSearching for .h5 files...")
    
    h5_files = list(Path("artifacts").rglob("*.h5"))
    if h5_files:
        print("Found:")
        for f in h5_files:
            print(f"  - {f}")
        print(f"\n💡 Update KERAS_MODEL_PATH in this script to one of the above")
    else:
        print("No .h5 files found! You need to re-train the model.")
    sys.exit(1)

# ========================================
# VERIFY MODEL ARCHITECTURE
# ========================================

print(f"\n[2/4] Verifying model architecture...")
print(f"   Input shape:  {model.input_shape}")
print(f"   Output shape: {model.output_shape}")

# Check if input shape is suitable for audio
if model.input_shape[1] is None or model.input_shape[1] < 8000:
    print(f"\n⚠️  WARNING: Model input shape {model.input_shape} seems wrong")
    print(f"   Expected shape like (None, 16000, 1) for audio")
    print(f"   This model might not be trained correctly")
    
    response = input("\nContinue anyway? (y/n): ")
    if response.lower() != 'y':
        sys.exit(1)
else:
    print(f"✅ Input shape looks correct for audio")

# ========================================
# CONVERT TO TFLITE
# ========================================

print(f"\n[3/4] Converting to TFLite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional: Enable quantization for smaller/faster model
# Uncomment to enable:
# print("   Enabling quantization...")
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

print(f"✅ Conversion successful")
print(f"   Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")

# Save
print(f"\nSaving to: {TFLITE_OUTPUT_PATH}")
Path(TFLITE_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

with open(TFLITE_OUTPUT_PATH, 'wb') as f:
    f.write(tflite_model)

print(f"✅ Saved successfully")

# ========================================
# VALIDATE CONVERSION
# ========================================

print(f"\n[4/4] Validating TFLite model...")

interpreter = tf.lite.Interpreter(model_path=TFLITE_OUTPUT_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"\n📥 TFLite Input Tensor:")
print(f"   Shape: {input_details[0]['shape']}")
print(f"   Type:  {input_details[0]['dtype']}")
print(f"   Name:  {input_details[0]['name']}")

print(f"\n📤 TFLite Output Tensor:")
print(f"   Shape: {output_details[0]['shape']}")
print(f"   Type:  {output_details[0]['dtype']}")
print(f"   Name:  {output_details[0]['name']}")

# Test inference
print(f"\n🧪 Testing inference...")
test_input = np.random.randn(*input_details[0]['shape']).astype(input_details[0]['dtype'])
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

print(f"✅ Test inference successful")
print(f"   Input shape:  {test_input.shape}")
print(f"   Output shape: {output.shape}")

# ========================================
# FINAL VALIDATION
# ========================================

print("\n" + "="*70)
tflite_input_shape = input_details[0]['shape']

if len(tflite_input_shape) >= 2 and tflite_input_shape[1] >= 8000:
    print("✅ SUCCESS! Model fixed correctly")
    print(f"   Input shape: {tflite_input_shape}")
    print("\n📋 Next steps:")
    print(f"   1. Validate: python scripts/validate_deployment.py")
    print(f"   2. Update inference script to use: {TFLITE_OUTPUT_PATH}")
    print(f"   3. Test with real audio data")
else:
    print("❌ FAILED! Input shape is still wrong")
    print(f"   Current: {tflite_input_shape}")
    print(f"   Expected: [1, 16000] or [1, 16000, 1]")
    print("\n💡 The Keras model itself may have wrong architecture")
    print("   Check train.py and verify the model input layer")

print("="*70)
