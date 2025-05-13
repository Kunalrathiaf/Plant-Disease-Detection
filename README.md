# AgriLeaf: Plant Disease Detection using Deep Learning

**AgriLeaf** is a deep learning-based image classification project built to detect plant diseases from leaf images. By leveraging CNNs and transfer learning, it offers an efficient solution for early disease diagnosis, helping farmers and agricultural specialists take proactive measures.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Model Results](#model-results)

---

## Overview

Agricultural losses due to undetected plant diseases can be devastating. **AgriLeaf** provides an automated solution to identify diseases using leaf images. With a trained CNN model, this project supports both custom training and real-time web-based predictions using a Streamlit interface.

## Key Features

- Plant disease classification using CNN
- Transfer learning with pretrained models (ResNet, VGG)
- Data augmentation for generalization
- Evaluation with confusion matrix and visualizations
- Streamlit app for interactive prediction

## Dataset

We use the **PlantVillage Dataset** from Kaggle, which contains over 54,000 images across 38 classes of healthy and diseased plant leaves.

Link: https://www.kaggle.com/datasets/emmarex/plantdisease

After downloading, place the extracted dataset in the `data/` directory.

---

## Technologies Used

- Python 3.7+
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Jupyter Notebook
- Streamlit

---

## Model Results

Accuracy: Achieved XX% test accuracy
Evaluation: Confusion matrix, precision, recall, F1-score
Prediction Examples: Visualized with image, predicted label, and probability
