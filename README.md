# Chest X-Ray Disease Classifier

This project uses a deep learning model (DenseNet121) to predict 14 thoracic conditions from frontal chest X-ray images.  

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

- Cardiomegaly: 0.797  
- Emphysema: 0.722  
- Effusion: 0.756  
- Hernia: 0.833  
- Infiltration: 0.643  
- Mass: 0.728  
- Nodule: 0.683  
- Atelectasis: 0.718  
- Pneumothorax: 0.758  
- Pleural Thickening: 0.669  
- Pneumonia: 0.615  
- Fibrosis: 0.744  
- Edema: 0.766  
- Consolidation: 0.698  

**Macro AUC:** 0.724

## Future Work

- Train the model for more epochs (30-40)
- Improve threshold selection for each condition  
- Grad-CAM visualization
