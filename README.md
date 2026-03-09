# Diabetic Retinopathy Detection
Deep Learning-Based Retinal Disease Classification with Explainable AI.

## Overview

This project implements a deep learning system to automatically detect diabetic retinopathy from retinal fundus images. Diabetic retinopathy is a diabetes-related eye disease that can lead to vision loss if not detected early.

The model uses Convolutional Neural Networks (CNNs) to classify retinal images and identify signs of the disease. To improve interpretability, Grad-CAM visualizations are generated to highlight the retinal regions that influenced the model’s predictions.

This project demonstrates how deep learning can assist in medical image analysis and early disease screening.

---

## Features

- CNN-based medical image classification
- Automated detection of diabetic retinopathy
- Trained on 1000+ retinal fundus images
- Image preprocessing and augmentation
- Grad-CAM heatmap visualization for explainability
- Visualization of important retinal regions affecting predictions

---

## Tech Stack

- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Matplotlib  
- Scikit-learn  

---

## Dataset

The model is trained using retinal fundus images labeled for diabetic retinopathy severity.

Dataset Link:  
https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions

Dataset characteristics:

- High-resolution retinal fundus images  
- Multiple stages of diabetic retinopathy  
- More than 1000 images used for training and testing  

---

## Project Structure

```
Diabetic-Retinopathy-Detection/
│
├── dataset/
│   ├── train/
│   ├── test/
│
├── models/
│   └── cnn_model.h5
│
├── notebooks/
│   └── training.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   └── gradcam_visualization.py
│
├── results/
│   ├── accuracy_plots.png
│   └── gradcam_outputs/
│
├── requirements.txt
└── README.md
```

---

## Model Architecture

The system uses a Convolutional Neural Network designed for image classification.

The architecture includes:

- Convolution layers for feature extraction  
- Max pooling layers for dimensionality reduction  
- Dropout layers to prevent overfitting  
- Fully connected layers for classification  
- Softmax activation for prediction

The CNN learns to identify retinal abnormalities such as:

- Microaneurysms  
- Hemorrhages  
- Exudates  
- Abnormal blood vessel patterns

---

## Explainable AI with Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to visualize which regions of the retinal image influenced the model’s prediction.

This helps:

- Understand model decision-making  
- Identify pathological retinal areas  
- Improve trust in AI-based medical systems

The output is a heatmap overlay on the retinal image highlighting important regions.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Diabetic-Retinopathy-Detection.git
cd Diabetic-Retinopathy-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Train the model:

```bash
python train_model.py
```

Run prediction on a new image:

```bash
python predict.py --image sample_retina.jpg
```

Generate Grad-CAM visualization:

```bash
python gradcam_visualization.py
```

---

## Results

The trained CNN is able to classify retinal images and detect patterns associated with diabetic retinopathy.

Outputs include:

- Model accuracy and loss curves  
- Predicted disease classification  
- Grad-CAM heatmaps highlighting critical retinal regions  

---

## Applications

- Automated retinal disease screening  
- AI-assisted ophthalmology diagnostics  
- Early detection of diabetic retinopathy  
- Clinical decision support systems  

---

## Future Improvements

- Train on larger datasets such as EyePACS  
- Improve accuracy using transfer learning models (ResNet, EfficientNet)  
- Deploy as a web application using Flask or FastAPI  
- Integrate real-time medical image analysis systems  

---
