#!/bin/bash
# File: check_dependencies.sh
# Purpose: Verify all required dependencies are installed

echo "=========================================="
echo "System Dependencies Check"
echo "=========================================="

# Python check
echo -e "\n[1/7] Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✅ $PYTHON_VERSION"
else
    echo "❌ Python3 not found"
fi

# TFLite Runtime
echo -e "\n[2/7] Checking TFLite Runtime..."
python3 -c "import tflite_runtime.interpreter; print('✅ tflite-runtime:', tflite_runtime.__version__)" 2>/dev/null || \
python3 -c "import tensorflow as tf; print('✅ tensorflow:', tf.__version__)" 2>/dev/null || \
echo "❌ Neither tflite-runtime nor tensorflow installed"

# NumPy
echo -e "\n[3/7] Checking NumPy..."
python3 -c "import numpy as np; print('✅ numpy:', np.__version__)" 2>/dev/null || \
echo "❌ NumPy not installed"

# Audio libraries
echo -e "\n[4/7] Checking Audio Libraries..."
python3 -c "import sounddevice as sd; print('✅ sounddevice:', sd.__version__)" 2>/dev/null || \
python3 -c "import pyaudio; print('✅ pyaudio:', pyaudio.__version__)" 2>/dev/null || \
echo "⚠️  No audio library found (install sounddevice or pyaudio)"

# SciPy (optional, for signal processing)
echo -e "\n[5/7] Checking SciPy (optional)..."
python3 -c "import scipy; print('✅ scipy:', scipy.__version__)" 2>/dev/null || \
echo "ℹ️  SciPy not installed (optional)"

# Check audio system (ALSA on Linux)
echo -e "\n[6/7] Checking Audio System..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if command -v aplay &> /dev/null; then
        echo "✅ ALSA tools installed"
        aplay -l 2>/dev/null | head -5
    else
        echo "⚠️  ALSA tools not found"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "ℹ️  macOS audio system (CoreAudio)"
else
    echo "ℹ️  Windows/Other OS"
fi

# Disk space check
echo -e "\n[7/7] Checking Disk Space..."
if command -v df &> /dev/null; then
    df -h . | tail -1 | awk '{print "✅ Available space:", $4}'
else
    echo "ℹ️  Cannot check disk space"
fi

echo -e "\n=========================================="
echo "Dependencies Check Complete"
echo "=========================================="
