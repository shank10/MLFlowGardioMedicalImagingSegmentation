# config.py
import os

# Directory paths
DATA_DIR = os.path.join(os.getcwd(), "data")
MODEL_DIR = os.path.join(os.getcwd(), "models")

# Training parameters
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
EPOCHS = 5
IMG_SIZE = (128, 128)

# Model and logging
MODEL_NAME = "unet"
MLFLOW_EXPERIMENT_NAME = "Medical Image Segmentation"
