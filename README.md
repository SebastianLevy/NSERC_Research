
# README

## Overview
This document provides instructions for using our research code, specifically focused on training and evaluating models for surface electromyography (sEMG) signal classification. The codebase includes scripts for data collection, processing, model training, and evaluation.

## Table of Contents
- [Data Collection](#data-collection)
- [Data Loading and Processing](#data-loading-and-processing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Model Explanations](#model-explanations)
- [Grasp Modes](#grasp-modes)

## Data Collection
Data is collected using an Arduino setup, which captures sEMG signals. The `dataCollection.py` script is used to read and store this data.

## Data Loading and Processing
### `datasetLoading.py`
- Load and preprocess sEMG data using the `sEMGDataset` class.
- `ComputeBasicFeatures` transform is applied for initial feature extraction.

## Model Training and Evaluation

### Prerequisites
- Python 3.x
- Libraries: PyTorch, NumPy, Pandas (complete list in `requirements.txt`)
- sEMG dataset in the specified format

### Setup
1. Install Python 3.x and all required libraries.
2. Clone the repository and navigate to the project directory.
3. Prepare your dataset according to the format used in `dataCollection.py`.

### Training
1. **Data Loading**: Utilize the `sEMGDataset` class from `datasetLoading.py` to load your dataset.
2. **Preprocessing**: Apply the `ComputeBasicFeatures` for initial data processing.
3. **Model Initialization**: Initialize the `HybridModel` from `modeling/hybridModel.py`.
4. **Training Configuration**: Set up the training environment using the Adam optimizer with a learning rate of 0.001.
5. **Execute Training**: Run the training loop for the desired number of epochs (default is 10).
6. **Monitoring**: Monitor training progress and performance on a validation set.

### Evaluation
- After training, evaluate the model's performance on a separate test dataset.
- Document the model's accuracy, precision, recall, and F1-score.

### Saving and Utilizing the Model
- Instructions for saving the trained model to a file.
- Guidelines for loading the model for future inference tasks.

## Model Explanations
### LSTM Model
- The LSTM model is designed to handle sequential data, capturing temporal dependencies in sEMG signals.
### Hybrid Model (CNN-LSTM)
- This model combines Convolutional Neural Networks (CNNs) for feature extraction with LSTM layers for sequence modeling, suitable for complex sEMG patterns.

## Grasp Modes
The system classifies sEMG signals into five grasp modes:
1. **Cylindrical Grasp**
2. **Tip Grasp**
3. **Hook Grasp**
4. **Palmar Grasp**
5. **Lateral Grasp**

Each mode has distinct applications in prosthetic control and robotics.

---

For further inquiries or support, please contact Sebastian Levy at sebastianlevy@cmu.edu

---

