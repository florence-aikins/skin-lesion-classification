# Deep Learning-Based Classification of Benign and Malignant Skin Lesions

A deep learning project that compares a custom Convolutional Neural Network (CNN), ResNet18 and EfficientNet-B0 for binary skin lesion classification using dermoscopic images.

## Project Overview

Skin cancer is one of the most common forms of cancer worldwide, and early diagnosis is essential for improving patient outcomes. This project investigates the use of deep learning models to automatically classify dermoscopic images as benign or malignant.

For this project, the classes were converted into binary labels:

### Binary Classes

| Label | Class |
|-------|--------|
| 0 | Benign |
| 1 | Malignant |


| Original Diagnosis | Label |
|-------------------|-------|
| Melanoma (mel) | Malignant (1) |
| Basal Cell Carcinoma (bcc) | Malignant (1) |
| Actinic Keratosis (akiec) | Malignant (1) |
| Benign Keratosis (bkl) | Benign (0) |
| Melanocytic Nevus (nv) | Benign (0) |
| Dermatofibroma (df) | Benign (0) |
| Vascular Lesions (vasc) | Benign (0) |


## Project Workflow

- Data loading
- Exploratory Data Analysis (EDA)
- Missing value treatment
- Image preprocessing
- Image resizing
- Data augmentation
- Stratified train-validation-test split
- Transfer learning
- Model training
- Model evaluation


## Exploratory Data Analysis

EDA included:
- Dataset overview
- Missing value analysis
- Age distribution
- Sex distribution
- Lesion localization
- Diagnosis distribution
- Benign vs malignant distribution
- Correlation heatmap
- Sample skin lesion images
- Image dimension analysis

Three models were developed and evaluated:

-  Custom CNN
-  ResNet18 (Transfer Learning)
-  EfficientNet-B0 (Transfer Learning)

The models were compared using multiple evaluation metrics to determine which architecture achieved the best classification performance.


## Dataset

**Dataset:** HAM10000 (Human Against Machine with 10,000 Training Images)
**Source:** https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000


### Dataset Split

- **70%** Training
- **15%** Validation
- **15%** Testing


## Data Preprocessing

The following preprocessing techniques were applied before model training:

- Resize images to **224 × 224**
- Random Horizontal Flip
- Random Vertical Flip
- Random Rotation
- Random Resized Crop
- Color Jitter
- ImageNet Normalization
- Class Weighting to address dataset imbalance


## Models Implemented

### Custom CNN

A CNN built from scratch consisting of:

- Convolutional Layers
- ReLU Activation
- MaxPooling
- Dropout
- Fully Connected Layers


### 2️⃣ ResNet18

Transfer learning using a pretrained ResNet18 model.

- ImageNet pretrained weights
- Final fully connected layer replaced with a 2-class classifier
- Fine-tuned on the HAM10000 dataset


### EfficientNet-B0

Transfer learning using EfficientNet-B0.

- ImageNet pretrained weights
- Fine-tuned on the HAM10000 dataset
- Produced the highest overall performance


## ⚙️ Training Configuration

| Parameter | Value |
|-----------|--------|
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | Weighted CrossEntropyLoss |
| Epochs | 15 |
| Framework | PyTorch |
| Environment | Google Colab (GPU) |


## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix


# Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|---------:|----------:|--------:|---------:|---------:|
| Custom CNN | 77.18% | 45.37% | 83.62% | 58.82% | 86.83%* |
| ResNet18 | 88.96% | 69.91% | 76.11% | 72.88% | 93.86% |
| **EfficientNet-B0** | **91.28%** | **78.13%** | **76.79%** | **77.45%** | **95.59%** |

> **Best Model:** EfficientNet-B0

# Key Findings

- EfficientNet-B0 achieved the highest overall performance with 91.28% accuracy.
- Transfer learning significantly outperformed the custom CNN.
- EfficientNet achieved the highest ROC-AUC (95.59%), demonstrating excellent discrimination between benign and malignant lesions.
- Weighted CrossEntropyLoss helped reduce the effect of class imbalance.
- Data augmentation improved model generalisation.

# Confusion Matrix Comparison

| Model | True Negative (TN) | False Positive (FP) | False Negative (FN) | True Positive (TP) |
|--------|-------------------:|--------------------:|--------------------:|-------------------:|
| Custom CNN | 915 | 295 | 48 | 2245 |
| ResNet18 | 1114 | 96 | 70 | 223 |
| EfficientNet-B0 | **1147** | **63** | **68** | **225** |

### Interpretation

- **True Negative (TN):** Benign lesions correctly classified as benign.
- **False Positive (FP):** Benign lesions incorrectly classified as malignant.
- **False Negative (FN):** Malignant lesions incorrectly classified as benign.
- **True Positive (TP):** Malignant lesions correctly classified as malignant.

  
# Technologies Used

- Python
- PyTorch
- Torchvision
- TorchMetrics
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab

# Running the Project

## Clone the repository

```bash
git clone https://github.com/florence-aikins/skin-lesion-classification.git
```

## Navigate into the project

```bash
cd skin-lesion-classification
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```
skin_lesion_project.ipynb
```

# Future Improvements

- Train on larger skin lesion datasets (e.g., ISIC 2020)
- Hyperparameter optimisation
- Vision Transformers (ViT)
- Grad-CAM visualisations
- Multi-class skin lesion classification
- Model deployment using Streamlit or Gradio


# Author

Florence Aikins
MSc Data Science
Manchester Metropolitan University

🔗 GitHub: https://github.com/florence-aikins

# Citation

Tschandl, P., Rosendahl, C., & Kittler, H. (2018). *The HAM10000 dataset: A large collection of multi-source dermatoscopic images of common pigmented skin lesions*. Scientific Data, 5, 180161.

## If you found this project useful, consider giving it a star!
