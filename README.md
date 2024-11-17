# Pneumonia Classification Using CNN with VGG19 and Flask

This project implements a Convolutional Neural Network (CNN) using a pre-trained VGG19 model for classifying chest X-ray images as either **Normal** or **Pneumonia**. The web application is built with Flask, allowing users to upload chest X-ray images and get predictions.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Model Performance](#Model-Performance)
- [Deployment with Flask](#Deployment-with-Flask)
- [Usage](#usage)


## Project Overview
The project involves:
1. Training a CNN model using VGG19 as the base architecture with transfer learning.
2. Augmenting the data to improve generalization and prevent overfitting.
3. Building a web interface using Flask for easy image uploads and predictions.

## Dataset
- Source: Chest X-Ray Images (Pneumonia) dataset on Kaggle
- Description: The dataset includes chest X-ray images categorized into two classes:
    - **PNEUMONIA** (positive cases)
    - **NORMAL** (healthy cases)
  
## Data Preprocessing and Augmentation
- Image resizing: Images are resized to (128, 128) for input into the model.
- Augmentation: Used for training data to improve generalization:
    - Horizontal and vertical flips
    - Random rotations and shifts
    - Rescaling pixel values to the [0, 1] range 

## Model Architecture
- **Base Model**: VGG19 pre-trained on ImageNet.
- **Custom Layers**:
    - Flatten() to convert 3D features to 1D
    - Fully connected layers with ReLU activation
    - Dropout layers for regularization
    - Final output layer with softmax activation for binary classification

- **Optimizer**: Stochastic Gradient Descent (SGD) with momentum.
- **Loss Function**: Categorical Crossentropy.
- **Metrics**: Accuracy
- **Callbacks**:
    - ModelCheckpoint for saving the best model
    - EarlyStopping for halting training when validation loss stops improving
    - ReduceLROnPlateau for learning rate adjustment

## Model Performance
- Validation Accuracy: 81.25%
- Test Accuracy: 91.51%

## Deployment with Flask
A simple web app is built using Flask to upload and predict images. The app:

1. Accepts an uploaded X-ray image.
2. Processes the image and runs it through the trained model.
3. Returns whether the image shows a Normal or Pneumonia result.

#### Run the Flask App
- python app.py
- Navigate to http://127.0.0.1:5000/ in your browser to use the web app.

## Usage
- Upload an X-ray image in the web interface.
- Click "Predict" to get the classification result.
- The app will display whether the uploaded image is classified as Normal or Pneumonia.
