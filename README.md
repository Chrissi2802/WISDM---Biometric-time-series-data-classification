# WISDM - Biometric time series data classification

This repository contains several models for a classification of the reduced WISDM dataset. <br>
Neural networks are used for feature extraction and classification. <br>
These were implemented in Python using the PyTorch library. <br>


## Data
The task is a classification of biometric time series data. The dataset is the "WISDM Smartphone and Smartwatch Activity and Biometrics Dataset", WISDM stands for Wireless Sensor Data Mining. The actual dataset was created by the Department of Computer and Information Science at Fordham University in New York. The researchers collected data from the accelerometer and gyroscope sensors of a smartphone and smartwatch as 51 subjects performed 18 diverse activities of daily living. Each activity was performed for 3 minutes, so that each subject contributed 54 minutes of data. <br> 
A detailed description of the dataset is also included in this repo. However, if you would like to view the original data, you can find the complete dataset here: https://www.cis.fordham.edu/wisdm/dataset.php <br>

As already mentioned, a reduced dataset is used, which contains the following six activities: <br>
A - walking <br>
B - jogging <br>
C - climbing stairs <br>
D - sitting <br>
E - standing <br>
M - kicking soccer ball <br>


## Models
Moreover, not only five different neural networks are available, but training procedures and data pre-processing scripts are also included. <br>

Models (neural networks): 
- Linear Model <br>
- CNN 1D Model <br>
- GRU (RNN) Model <br>
- CNN 2D Model <br>
- LSTM (RNN) Model <br>

datasets.py provides the data set and prepares the data <br>
helpers.py provides auxiliary classes and functions for neural networks <br>
models.py provides the models <br>
train.py trains the model <br>
