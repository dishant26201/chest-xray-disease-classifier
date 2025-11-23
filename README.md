# Chest X-Ray Disease Classifier

This project uses a deep learning model (DenseNet121) to predict 14 thoracic conditions from frontal chest X-ray images.

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

Training was planned for 15 epochs, but was very time consuming so it was manually stopped early at around 12 epochs (11 fully completed).
Longer training would likely improve results.

## Evaluation (ROC–AUC)

Per-class ROC–AUC scores:

- Cardiomegaly: 0.845
- Emphysema: 0.805
- Effusion: 0.801
- Hernia: 0.846
- Infiltration: 0.678
- Mass: 0.782
- Nodule: 0.729
- Atelectasis: 0.739
- Pneumothorax: 0.796
- Pleural Thickening: 0.708
- Pneumonia: 0.680
- Fibrosis: 0.780
- Edema: 0.806
- Consolidation: 0.714

**Macro AUC:** 0.765

## Future Work

- Train the model for more epochs (30-40)
- Improve threshold selection for each condition
- Grad-CAM visualization
