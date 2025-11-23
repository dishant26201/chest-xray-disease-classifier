# src/preprocessing.py

import numpy as np
import tensorflow as tf
from .config import MODEL_PATH, IMG_SIZE, LABELS

EPS = np.finfo(float).eps  # Machine epsilon to prevent divide by zero error

# Preprocess an image for the chest X-ray model
def preprocess_image(input_image):
    image = input_image.convert("RGB") # Ensure image is in RGB format

    image = image.resize(IMG_SIZE) # Resize to the image size used during training

    image_arr = np.array(image).astype("float32") # Convert to NumPy array (H, W, 3)

    # Samplewise centering by subtracting mean of this image
    mean = image_arr.mean()
    image_arr = image_arr - mean

    # Samplewise std normalization by dividing by std of this image
    std = image_arr.std() + EPS  # Machine epsilon to avoid division by zero
    image_arr = image_arr / std

    image_batch = image_arr.reshape(1, *image_arr.shape) # Add batch dimension to make shape (1, H, W, 3)


    return image_batch

# Run the model on the given image and return probabilities for each label
def predict_labels(input_image, model):
    image_batch = preprocess_image(input_image) # Preprocess the image

    preds = model.predict(image_batch)[0] # Predicted probabilities 

    preds_dict = {label: float(pred) for label, pred in zip(LABELS, preds)} # Map output probabilities to label names

    return preds_dict # Return the dictionary of the predicted probabilities
