# Fixing TFLite Model Shape Error

## 🚨 Problem Identified

**Validation Results**: Your TFLite model has an incorrect input shape.

### Root Cause in `train.py`

**Line 159** in `train.py` defines the model with a flexible input shape:

```python
@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
def call(self, waveform):
```

The `shape=[None]` means "accept any length waveform", but when converting to TFLite, this gets collapsed to `shape=[1]` because TFLite requires concrete shapes.

**The Fix**: Change `shape=[None]` to `shape=[16000]` to specify the exact audio length.

```
❌ FAIL: Input Shape for Audio Data
   Current shape: [1]
   Expected shape: [1, 16000] or [1, 16000, 1]
```

**Error in production**:
```
ValueError: Cannot set tensor: Dimension mismatch. 
Got 16000 but expected 1 for dimension 0 of input
```

**Root Cause**: The Keras model was converted to TFLite incorrectly, losing the audio input dimensions.

---

## ✅ Solution: Re-export the Model

### Step 1: Locate Original Keras Model

```bash
# Search for .h5 files in artifacts directory
cd /home/pi/Noise_Anomaly_deployment/Sound-Anamoly-Transfer-Learning
find artifacts/ -name "*.h5"

# Expected output:
# artifacts/sound_classifier_head.h5
# OR
# artifacts/2025_XX_XX_XX_XX_XX/sound_classifier_head.h5
```

---

### Step 2: Create Fix Script

Create a file: `scripts/fix_tflite_export.py`

```python
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
```

---

### Step 3: Run the Fix Script

```bash
# Make sure you're in the project root
cd /home/pi/Noise_Anomaly_deployment/Sound-Anamoly-Transfer-Learning

# Run the fix script
python scripts/fix_tflite_export.py
```

**Expected output**:
```
✅ SUCCESS! Model fixed correctly
   Input shape: [1, 16000, 1]
```

---

### Step 4: Validate the Fixed Model

```bash
# Point to the new fixed model
export TFLITE_MODEL_PATH=artifacts/sound_classifier_fixed.tflite

# Run validation
python scripts/validate_deployment.py
```

**You should now see**:
```
[1/5] Input Shape for Audio Data
  ✅ PASS: [1, 16000, 1] suitable for audio (16000 samples)

...

✅ ALL CHECKS PASSED - Model ready for deployment!
```

---

### Step 5: Update Inference Script

Update your inference script to use the fixed model:

```python
# OLD (broken model)
MODEL_PATH = "/app/test/artifacts/sound_classifier.tflite"

# NEW (fixed model)
MODEL_PATH = "/app/test/artifacts/sound_classifier_fixed.tflite"
```

---

## 🔍 Troubleshooting

### Issue: Can't Find .h5 File

**Solution**: You need to re-train the model

```bash
# Run the training pipeline
./run_pipeline.sh --data ./data

# This will create a new .h5 file in artifacts/
```

---

### Issue: Keras Model Has Wrong Shape

If the Keras model itself has shape `(None, 1)` or similar, the model was trained incorrectly.

**Check `train.py`**:
```python
# BAD - Wrong input shape
model.add(Input(shape=(1,)))

# GOOD - Correct for audio
model.add(Input(shape=(16000, 1)))
```

**Solution**: Fix the model architecture in `train.py` and re-train.

---

### Issue: Still Getting Shape Errors

If the fixed model still has issues:

1. **Verify the Keras model architecture**:
   ```python
   import tensorflow as tf
   model = tf.keras.models.load_model('artifacts/sound_classifier_head.h5')
   model.summary()  # Check the first layer
   ```

2. **Check if using the correct model file**:
   ```bash
   # List all .h5 files
   find artifacts/ -name "*.h5" -exec ls -lh {} \;
   ```

3. **Ensure TensorFlow version matches**:
   ```bash
   # Check TensorFlow version
   python -c "import tensorflow as tf; print(tf.__version__)"
   
   # Should be >= 2.10.0
   ```

---

## 📊 Complete Workflow

```bash
# 1. Find Keras model
find artifacts/ -name "*.h5"

# 2. Update fix script with correct path
nano scripts/fix_tflite_export.py
# Update: KERAS_MODEL_PATH = "artifacts/YOUR_MODEL.h5"

# 3. Run fix
python scripts/fix_tflite_export.py

# 4. Validate
export TFLITE_MODEL_PATH=artifacts/sound_classifier_fixed.tflite
python scripts/validate_deployment.py

# 5. Update inference script
nano inference.py  # or your inference script
# Change MODEL_PATH to point to fixed model

# 6. Test inference
python inference.py --model artifacts/sound_classifier_fixed.tflite
```

---

## ✅ Success Criteria

Your model is fixed when:

- ✅ Validation shows input shape `[1, 16000]` or `[1, 16000, 1]`
- ✅ All 5 validation checks pass
- ✅ End-to-end inference test succeeds
- ✅ Actual inference works without dimension errors

---

## 📚 Related Documentation

- **Validation Guide**: `DEPLOYMENT_DEBUG_GUIDE.md` - Section 1
- **Validation Script**: `scripts/validate_deployment.py`
- **Quick Diagnosis**: `scripts/quick_diagnosis.sh`

---

**Last Updated**: 2026-01-16  
**Validated On**: Raspberry Pi (Python 3.11.9, tflite-runtime)
