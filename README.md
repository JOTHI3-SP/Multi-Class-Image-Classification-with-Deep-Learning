# Multi-Class-Image-Classification-with-Deep-Learning

# CIFAR-10 Image Classification 

## Project Overview
This project implements a **deep learning–based image classification pipeline** using the **CIFAR‑10 dataset**. The notebook demonstrates the complete workflow of loading image data, preprocessing inputs, building a neural network model, training it on labeled data, and evaluating its classification performance.

The task is formulated as a **10‑class supervised learning problem**.

## Dataset Details
- Dataset: CIFAR‑10
- Total images: 60,000
- Training samples: 50,000
- Test samples: 10,000
- Image resolution: 32 × 32 (RGB)
- Number of classes: 10

Class labels:
Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

## Data Preprocessing
- Dataset loaded using deep learning utilities
- Pixel values normalized to improve numerical stability during training
- Images reshaped to match model input requirements
- Labels prepared for multi‑class classification

## Model Design
- Model Type: **Deep Learning Neural Network**
- Learning Paradigm: **Supervised Learning**
- Input: 32 × 32 × 3 RGB images
- Output: Probability distribution over 10 classes
- Dense layers used for feature learning
- Softmax activation applied at the output layer

## Training Configuration
- Loss Function: Categorical Cross‑Entropy
- Optimizer: Gradient‑based optimizer
- Training performed over multiple epochs
- Model trained on CIFAR‑10 training set
- Validation performed using test data

## Performance Evaluation
- Training and validation accuracy monitored during learning
- Final test accuracy reported in the notebook
- Learning curves visualized to analyze convergence behavior

## Results
The trained model is able to learn discriminative visual patterns from CIFAR‑10 images and correctly classify images across multiple object categories, demonstrating the application of deep learning techniques to multi‑class image recognition.

## Tools and Libraries
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Jupyter Notebook

## How to Run
1. Clone the repository
2. Install dependencies:
   pip install tensorflow numpy matplotlib
3. Open the notebook
4. Execute cells sequentially to reproduce results

## Summary
This project presents a structured implementation of an image classification system using deep learning, covering data preprocessing, model construction, training, and evaluation on a standard benchmark dataset.
