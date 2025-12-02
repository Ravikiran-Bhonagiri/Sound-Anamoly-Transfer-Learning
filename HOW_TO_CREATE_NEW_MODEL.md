# Workflow: Retraining the Model with New Data

This guide outlines the steps to update the sound anomaly detection model when you have collected new audio data.

## 1. Prepare the Data
1.  **Collect Audio**: Ensure your new audio recordings are in `.wav` format.
2.  **Organize Files**: Place the new files into the corresponding class folders in the `data/` directory:
    *   `data/on_state/`
    *   `data/off_state/`
    *   `data/solid_state/`
    *   `data/soft_state/`
3.  **Clean Up**: Remove any bad recordings or files that are too short (< 1 second).

## 2. Verify Data Quality (Optional but Recommended)
Before training, it's good practice to visualize the new data to ensure it clusters well.

1.  Run the visualization script:
    ```bash
    python visualize_data.py
    ```
2.  Check the output plot: `artifacts/embeddings_pca_check.png`.
3.  **Success Criteria**: You should see distinct clusters for each class. If classes overlap significantly, you may need more data or better recording quality.

## 3. Retrain the Model
Run the training script to generate a new model based on the updated dataset.

1.  Execute the training script:
    ```bash
    python train.py
    ```
2.  **Outputs**: This will overwrite the following files in the `artifacts/` folder:
    *   `sound_classifier.tflite` (The deployed model)
    *   `sound_classifier_head.h5` (The Keras model)

## 4. Evaluate Performance
Check the metrics to ensure the new model performs well.

1.  Run the evaluation script:
    ```bash
    python evaluate_model.py
    ```
2.  **Check Results**: Look at the console output for Accuracy/F1-Score and check `artifacts/confusion_matrix.png`.

## 5. Update the Docker Image
To deploy the new model to the Raspberry Pi, you must rebuild the Docker image.

1.  Rebuild the image:
    ```bash
    docker build -t sound_classifier .
    ```
    *Note: The Dockerfile automatically copies the updated `artifacts/sound_classifier.tflite` into the image.*

## 6. Verify and Deploy
1.  **Test Locally**:
    ```bash
    # Replace with a path to one of your new files
    docker run --rm -v "${PWD}:/app/test" sound_classifier python /app/test/test_docker_inference.py /app/test/data/on_state/new_file.wav
    ```
2.  **Deploy**:
    *   If running on the same machine, just run the new image.
    *   If deploying to a fleet, push the new image to your container registry (e.g., Docker Hub) and pull it on the Raspberry Pi.
