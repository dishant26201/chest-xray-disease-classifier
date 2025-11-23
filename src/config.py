# src/config.py

from pathlib import Path

MODEL_PATH = Path("models/chest_xray_classifier_v3_best.h5") # Path to the saved Keras model

IMG_SIZE = (320, 320)  # Image size used for training DenseNet121 model (width, height)

LAST_CONV_LAYER = "conv5_block16_concat"

# List of labels in the exact order the model outputs them
LABELS = [
    "Cardiomegaly", 
    "Emphysema", 
    "Effusion", 
    "Hernia", 
    "Infiltration", 
    "Mass", 
    "Nodule", 
    "Atelectasis", 
    "Pneumothorax", 
    "Pleural_Thickening", 
    "Pneumonia", 
    "Fibrosis", 
    "Edema", 
    "Consolidation"
]