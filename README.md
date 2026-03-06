# cnn-rotor-signal-classification
This project implements a 1D Convolutional Neural Network (CNN) to classify rotor pressure signals using TensorFlow/Keras. The dataset is imbalanced, so class weights and stratified K-fold cross-validation are used to obtain robust evaluation.

## Methods
- 1D CNN
- Stratified K-Fold Cross Validation
- Class Imbalance Handling (class weights)
- Early Stopping
- Learning Rate Scheduling
- Model Checkpointing
- Evaluation using F1-score and MCC
- 
- ## Technologies
- Python
- TensorFlow / Keras
- NumPy
- Scikit-learn
- Matplotlib

- ## Environment
- Python 3.12
- TensorFlow 2.19
- Keras 3.11
- NumPy 2.1

- ## Results
Mean F1-score: 0.95 ± 0.05  
Mean MCC: 0.95 ± 0.05

## How to Run
1. Install dependencies:
   pip install -r requirements.txt

2. Train the model:
   python train_model.py
