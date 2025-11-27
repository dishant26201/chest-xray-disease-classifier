# Chest X-Ray Disease Classifier

This project uses a deep learning model (DenseNet121) to predict 14 thoracic conditions from frontal chest X-ray images.

**Note:** I continue to improve and refine this project from time to time as I learn more about sleep analysis and EEG processing.
**Try the live demo:** https://chest-xray-disease-classifier-demo.streamlit.app/

## Model Development

- Training was performed in a Kaggle notebook using the [NIH ChestX-ray14 dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC).
- Images were resized to 320×320 and normalized using sample-wise centering and standardization.
- Random horizontal flipping was applied for data augmentation.
- The model architecture included:
  - DenseNet121 pretrained on Imagenet weights
  - Global Average Pooling
  - A final dense layer with 14 sigmoid outputs for multi-label prediction
- Additional steps:
  - Patient-wise splitting to prevent data leakage
  - Handling class imbalance using weighted binary cross-entropy
  - Early stopping during training and saving best versions during training

## Limitations

Training was planned for 50 epochs, but was very time consuming so it was manually stopped early at around 20 epochs.
Longer training would likely improve results.

## Evaluation (ROC–AUC)

Per-class ROC–AUC scores:

- Cardiomegaly: 0.89
- Emphysema: 0.89
- Pneumothorax: 0.86
- Edema: 0.84
- Hernia: 0.84
- Mass: 0.83
- Effusion: 0.83
- Fibrosis: 0.82
- Nodule: 0.77
- Atelectasis: 0.77
- Consolidation: 0.75
- Pleural Thickening: 0.73
- Pneumonia: 0.72
- Infiltration: 0.71

**Macro AUC:** 0.80

## Future Work

- Train the model for more epochs (50-100)
- Improve threshold selection for each condition
