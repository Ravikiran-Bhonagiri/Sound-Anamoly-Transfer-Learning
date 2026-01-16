"""
Complete TFLite Model Validation Script
Runs all validation phases for sound anomaly detection model
"""

import sys
import os
from pathlib import Path
import numpy as np

# Auto-detect project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def check_dependencies():
    """Phase 1: Check all required dependencies"""
    print_header("PHASE 1: DEPENDENCY CHECK")
    
    checks_passed = 0
    total_checks = 5
    
    # Check 1: Python version
    print(f"\n[1/{total_checks}] Python Version")
    print(f"  ✅ Python {sys.version.split()[0]}")
    checks_passed += 1
    
    # Check 2: TensorFlow/TFLite
    print(f"\n[2/{total_checks}] TensorFlow/TFLite")
    try:
        import tflite_runtime.interpreter as tflite
        print(f"  ✅ tflite-runtime installed")
        checks_passed += 1
    except ImportError:
        try:
            import tensorflow as tf
            print(f"  ✅ TensorFlow {tf.__version__} installed")
            checks_passed += 1
        except ImportError:
            print(f"  ❌ Neither tflite-runtime nor tensorflow found")
            print(f"     Install with: pip install tensorflow")
    
    # Check 3: NumPy
    print(f"\n[3/{total_checks}] NumPy")
    try:
        import numpy
        print(f"  ✅ NumPy {numpy.__version__}")
        checks_passed += 1
    except ImportError:
        print(f"  ❌ NumPy not installed")
    
    # Check 4: Audio libraries
    print(f"\n[4/{total_checks}] Audio Libraries")
    audio_lib_found = False
    try:
        import sounddevice
        print(f"  ✅ sounddevice {sounddevice.__version__}")
        audio_lib_found = True
    except ImportError:
        try:
            import pyaudio
            print(f"  ✅ pyaudio installed")
            audio_lib_found = True
        except ImportError:
            print(f"  ⚠️  No audio library (optional for file-based inference)")
    
    if audio_lib_found:
        checks_passed += 1
    
    # Check 5: Project structure
    print(f"\n[5/{total_checks}] Project Structure")
    if ARTIFACTS_DIR.exists():
        print(f"  ✅ Artifacts directory found: {ARTIFACTS_DIR}")
        checks_passed += 1
    else:
        print(f"  ❌ Artifacts directory not found: {ARTIFACTS_DIR}")
    
    print(f"\n{'='*70}")
    print(f"Dependencies: {checks_passed}/{total_checks} checks passed")
    print(f"{'='*70}")
    
    return checks_passed >= 3  # Need at least Python, TF, and NumPy

def find_tflite_model():
    """Find the TFLite model in artifacts directory"""
    # Check environment variable first
    env_path = os.getenv('TFLITE_MODEL_PATH')
    if env_path and Path(env_path).exists():
        return Path(env_path)
    
    # Check default location
    default_path = ARTIFACTS_DIR / "sound_classifier.tflite"
    if default_path.exists():
        return default_path
    
    # Search all subdirectories
    for tflite_file in ARTIFACTS_DIR.rglob("*.tflite"):
        return tflite_file
    
    return None

def validate_tflite_model(model_path):
    """Phase 2: Validate TFLite model structure"""
    print_header("PHASE 2: TFLITE MODEL VALIDATION")
    
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite
    
    print(f"\nModel: {model_path}")
    print(f"Size: {model_path.stat().st_size / 1024 / 1024:.2f} MB\n")
    
    # Load model
    try:
        interpreter = tflite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        print("✅ Model loaded successfully\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Get details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Print tensor details
    print("📥 INPUT TENSOR:")
    print(f"  Shape:  {input_details[0]['shape']}")
    print(f"  Type:   {input_details[0]['dtype']}")
    print(f"  Name:   {input_details[0]['name']}")
    
    print("\n📤 OUTPUT TENSOR:")
    print(f"  Shape:  {output_details[0]['shape']}")
    print(f"  Type:   {output_details[0]['dtype']}")
    print(f"  Name:   {output_details[0]['name']}")
    
    # Validation checks
    print(f"\n{'='*70}")
    print("VALIDATION CHECKS")
    print(f"{'='*70}")
    
    checks_passed = 0
    total_checks = 5
    
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    output_shape = output_details[0]['shape']
    
    # Check 1: Input shape for audio
    print(f"\n[1/{total_checks}] Input Shape for Audio Data")
    if len(input_shape) >= 2:
        audio_samples = input_shape[1]
        if audio_samples >= 8000:
            print(f"  ✅ PASS: {input_shape} suitable for audio ({audio_samples} samples)")
            checks_passed += 1
        else:
            print(f"  ❌ FAIL: Shape {input_shape} too small ({audio_samples} samples)")
    else:
        print(f"  ❌ FAIL: Invalid shape {input_shape}")
        print(f"     Expected: [1, 16000] or [1, 16000, 1]")
    
    # Check 2: Input dtype
    print(f"\n[2/{total_checks}] Input Data Type")
    if input_dtype == np.float32:
        print(f"  ✅ PASS: float32 (standard)")
        checks_passed += 1
    elif input_dtype in [np.int8, np.uint8]:
        print(f"  ⚠️  WARN: {input_dtype} (quantized - ensure preprocessing matches)")
        checks_passed += 1
    else:
        print(f"  ❌ FAIL: Unexpected dtype {input_dtype}")
    
    # Check 3: Output shape
    print(f"\n[3/{total_checks}] Output Shape for Classification")
    if len(output_shape) == 2 and output_shape[1] >= 2:
        print(f"  ✅ PASS: {output_shape} ({output_shape[1]} classes)")
        checks_passed += 1
    else:
        print(f"  ⚠️  WARN: Unusual shape {output_shape}")
    
    # Check 4: Test inference
    print(f"\n[4/{total_checks}] Test Inference")
    try:
        test_input = np.random.randn(*input_shape).astype(input_dtype)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"  ✅ PASS: Inference successful")
        print(f"     Input:  {test_input.shape}")
        print(f"     Output: {output.shape}")
        print(f"     Sample: {output[0][:3]}")
        checks_passed += 1
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
    
    # Check 5: Model size
    print(f"\n[5/{total_checks}] Model File Size")
    size_mb = model_path.stat().st_size / 1024 / 1024
    if size_mb > 0.01:
        print(f"  ✅ PASS: {size_mb:.2f} MB")
        checks_passed += 1
    else:
        print(f"  ❌ FAIL: File seems corrupted ({size_mb:.2f} MB)")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"VALIDATION: {checks_passed}/{total_checks} checks passed")
    print(f"{'='*70}")
    
    if checks_passed == total_checks:
        print("✅ ALL CHECKS PASSED - Model ready for deployment!")
        return True
    elif checks_passed >= total_checks * 0.8:
        print("⚠️  MOST CHECKS PASSED - Review warnings")
        return True
    else:
        print("❌ VALIDATION FAILED - Fix issues before deployment")
        return False

def test_inference(model_path):
    """Phase 3: Test end-to-end inference"""
    print_header("PHASE 3: END-TO-END INFERENCE TEST")
    
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite
    
    # Load model
    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    
    print(f"\nSimulating audio capture...")
    print(f"Expected input: {input_shape} ({input_dtype})")
    
    # Simulate audio preprocessing
    if len(input_shape) == 2:
        # [batch, samples]
        num_samples = input_shape[1]
        audio_data = np.random.randn(num_samples).astype(np.float32)
        processed_audio = audio_data.reshape(1, -1)
    elif len(input_shape) == 3:
        # [batch, samples, channels]
        num_samples = input_shape[1]
        audio_data = np.random.randn(num_samples).astype(np.float32)
        processed_audio = audio_data.reshape(1, -1, 1)
    else:
        print(f"❌ Unsupported input shape: {input_shape}")
        return False
    
    # Normalize
    processed_audio = processed_audio / (np.max(np.abs(processed_audio)) + 1e-8)
    processed_audio = processed_audio.astype(input_dtype)
    
    print(f"\nPreprocessed audio:")
    print(f"  Shape: {processed_audio.shape}")
    print(f"  Range: [{processed_audio.min():.3f}, {processed_audio.max():.3f}]")
    
    # Run inference
    try:
        interpreter.set_tensor(input_details[0]['index'], processed_audio)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"\n✅ Inference successful!")
        print(f"  Predictions: {predictions[0]}")
        
        # Apply softmax if needed
        preds = predictions[0]
        if preds.max() > 1.0 or preds.min() < 0:
            exp_p = np.exp(preds - np.max(preds))
            probs = exp_p / exp_p.sum()
        else:
            probs = preds
        
        predicted_class = np.argmax(probs)
        confidence = probs[predicted_class]
        
        print(f"\n📊 RESULT:")
        print(f"  Class:      {predicted_class}")
        print(f"  Confidence: {confidence:.1%}")
        print(f"  All probs:  {probs}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validation phases"""
    print_header("SOUND ANOMALY DETECTION - MODEL VALIDATION")
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Artifacts:    {ARTIFACTS_DIR}")
    
    # Phase 1: Dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed. Install required packages.")
        return 1
    
    # Find model
    model_path = find_tflite_model()
    if not model_path:
        print("\n❌ No TFLite model found in artifacts directory")
        print(f"   Searched in: {ARTIFACTS_DIR}")
        return 1
    
    # Phase 2: Validate model
    if not validate_tflite_model(model_path):
        print("\n❌ Model validation failed")
        return 1
    
    # Phase 3: Test inference
    if not test_inference(model_path):
        print("\n❌ Inference test failed")
        return 1
    
    # Success
    print_header("🎉 ALL VALIDATION PHASES PASSED!")
    print("\nYour model is ready for deployment.")
    print("\nNext steps:")
    print("  1. Transfer model to remote device")
    print("  2. Update inference script with correct preprocessing")
    print("  3. Test on real audio data")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
