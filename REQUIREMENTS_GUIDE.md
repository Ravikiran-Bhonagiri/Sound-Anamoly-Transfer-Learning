# Project Requirements Guide

This document explains the different environment requirement files used in this project.

## 1. Training Environment (`requirements_training.txt`)
**Use this file on your PC / Laptop / Cloud Server.**

This file contains the full set of heavy libraries needed to:
*   Train the model (`train.py`)
*   Visualize data (`visualize_data.py`)
*   Evaluate performance (`evaluate_model.py`)

**Key Dependencies:**
*   `tensorflow`: The full deep learning framework.
*   `tensorflow-hub`: For loading the YAMNet model.
*   `scikit-learn`, `matplotlib`, `seaborn`: For data analysis and plotting.
*   `flatbuffers<2`: Pinned to ensure compatibility if you ever need to convert models in this environment.

**Installation:**
```bash
pip install -r requirements_training.txt
```

---

## 2. Inference Environment (`requirements_inference_pi.txt`)
**Use this file on your Raspberry Pi.**

This file is optimized for the edge device. It contains *only* the lightweight libraries needed to run the trained model in real-time.

**Key Dependencies:**
*   `tflite-runtime`: A much smaller version of TensorFlow just for running models.
*   `sounddevice`: For capturing audio from the microphone.
*   `numpy<2`: Pinned for compatibility with `tflite-runtime`.

**Installation (on Raspberry Pi):**
```bash
pip install -r requirements_inference_pi.txt
```
