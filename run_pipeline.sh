#!/bin/bash

# Default Configuration
IMAGE_TRAINING="sound-anomaly-training"
IMAGE_INFERENCE="sound-anomaly-inference"
BASE_DIR="$(pwd)"
DATA_DIR="$BASE_DIR/data"
ARTIFACTS_DIR="$BASE_DIR/artifacts"
TEST_FILE="$DATA_DIR/on_state/on_merged.wav"

# Help Function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -d, --data DIR       Path to training data directory (default: ./data)"
    echo "  -t, --test FILE      Path to test audio file (default: ./data/on_state/on_merged.wav)"
    echo "  -a, --artifacts DIR  Path to artifacts output directory (default: ./artifacts)"
    echo "  -h, --help           Show this help message"
    exit 1
}

# Parse Arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -d|--data) DATA_DIR="$2"; shift ;;
        -t|--test) TEST_FILE="$2"; shift ;;
        -a|--artifacts) ARTIFACTS_DIR="$2"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Convert to absolute paths
DATA_DIR=$(realpath "$DATA_DIR")
TEST_FILE=$(realpath "$TEST_FILE")
ARTIFACTS_DIR=$(realpath "$ARTIFACTS_DIR")

# Ensure directories exist
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found at $DATA_DIR"
    exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
    echo "Error: Test file not found at $TEST_FILE"
    exit 1
fi

if [ ! -d "$ARTIFACTS_DIR" ]; then
    mkdir -p "$ARTIFACTS_DIR"
fi

echo "=================================================="
echo "Configuration:"
echo "  Data Dir:      $DATA_DIR"
echo "  Test File:     $TEST_FILE"
echo "  Artifacts Dir: $ARTIFACTS_DIR"
echo "=================================================="

echo "Step 1: Running Training..."
# Run training container
docker run --rm \
    -v "$DATA_DIR:/app/data" \
    -v "$ARTIFACTS_DIR:/app/artifacts" \
    "$IMAGE_TRAINING"

if [ $? -ne 0 ]; then
    echo "Training failed."
    exit 1
fi

echo "=================================================="
echo "Step 2: Locating New Model..."
# Find the most recently created directory in artifacts
LATEST_ARTIFACT_DIR=$(ls -td "$ARTIFACTS_DIR"/*/ | head -1)

if [ -z "$LATEST_ARTIFACT_DIR" ]; then
    echo "Error: No artifact directory found."
    exit 1
fi

LATEST_ARTIFACT_DIR=${LATEST_ARTIFACT_DIR%/}
MODEL_PATH_HOST="$LATEST_ARTIFACT_DIR/sound_classifier.tflite"

if [ ! -f "$MODEL_PATH_HOST" ]; then
    echo "Error: Model file not found at $MODEL_PATH_HOST"
    exit 1
fi

echo "Found latest model: $MODEL_PATH_HOST"

echo "=================================================="
echo "Step 3: Running Inference Verification..."

# Mount the specific test file directory to allow access
TEST_FILE_DIR=$(dirname "$TEST_FILE")
TEST_FILE_NAME=$(basename "$TEST_FILE")

docker run --rm \
    -v "$TEST_FILE_DIR:/app/test_data" \
    -v "$MODEL_PATH_HOST:/app/model.tflite" \
    --entrypoint python \
    "$IMAGE_INFERENCE" \
    /app/test/test_docker_inference.py \
    "/app/test_data/$TEST_FILE_NAME" \
    "/app/model.tflite"

echo "=================================================="
echo "Pipeline Complete."
echo "=================================================="
