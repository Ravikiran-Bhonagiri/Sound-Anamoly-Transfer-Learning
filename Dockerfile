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
