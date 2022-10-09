# WISDM - Biometric time series data classification

This repository contains several models for a classification of the reduced WISDM dataset. <br>
Neural networks are used for feature extraction and classification. <br>
These were implemented in Python using the [PyTorch](https://pytorch.org/ "PyTorch website") library. The latest neural networks have been implemented in [TensorFlow](https://www.tensorflow.org/ "TensorFlow website"). All files or folders with a "_tf" or "_TF" in the name are for TensorFlow. <br>
This repository is based on a Kaggle Competition. The website for this Competition can be found [here](https://www.kaggle.com/competitions/anncmss22/overview "Kaggle Competition ANNs and Cognitive Models (FHWS SS22) website"). <br> 


## Data
The task is a classification of biometric time series data. The dataset is the "WISDM Smartphone and Smartwatch Activity and Biometrics Dataset", WISDM stands for Wireless Sensor Data Mining. The actual dataset was created by the Department of Computer and Information Science at Fordham University in New York. The researchers collected data from the accelerometer and gyroscope sensors of a smartphone and smartwatch as 51 subjects performed 18 diverse activities of daily living. Each activity was performed for 3 minutes, so that each subject contributed 54 minutes of data. <br> 
A detailed description of the dataset is also included in this repo. However, if you would like to view the original data, you can find the complete dataset [here](https://www.cis.fordham.edu/wisdm/dataset.php "WISDM Dataset website"). <br>

As already mentioned, a reduced dataset is used, which contains the following six activities: <br>
A - walking <br>
B - jogging <br>
C - climbing stairs <br>
D - sitting <br>
E - standing <br>
M - kicking soccer ball <br>


## Models
Moreover, not only eleven different neural networks are available, but training procedures and data pre-processing scripts are also included. <br>

Models (neural networks): 
- PyTorch
    - Linear / Multilayer Perceptron (MLP) model <br>
    - Convolutional Neural Network (CNN) 1D model <br>
    - Gated Recurrent Units (GRU), this is a Recurrent Neural Network (RNN) model <br>
    - CNN 2D model <br>
    - Long Short-Term Memory (LSTM) model <br>
- TensorFlow
    - MLP model <br> 
    - CNN 2D model <br>
    - GRU model <br>
    - LSTM model <br>
    - Big GRU model <br> 
    - Convolutional LSTM model <br> 


## Overview of the folder structure and files
| Files                           | Description                                                         |
| ------------------------------- | ------------------------------------------------------------------- |
| Datasets/                       | contains the data and the submissions                               |
| Models/                         | contains the trained models                                         |
| Plots/                          | contains all plots from the training and testing                    |
| .gitignore                      | contains files and folders that are not tracked via git             |
| dataset_tf.py                   | provides the dataset and prepares the data for TensorFlow           |
| datasets.py                     | provides the dataset and prepares the data for PyTorch              |
| helpers.py                      | provides auxiliary classes and functions for neural networks        |
| Job.sh                          | provides a script to carry out the training on a computer cluster   |
| models_tf.py                    | provides the models for TensorFlow                                  |
| models.py                       | provides the models for PyTorch                                     |
| train_tf.py                     | provides functions for training and testing for TensorFlow          |
| train.py                        | provides functions for training and testing for PyTorch             |
| WISDM-dataset-description.pdf   | further description of the dataset                                  |

## Achieved results
The scores were calculated by Kaggle. The metric is the categorization accuracy (ACC). <br> 
| Models             | Public leaderboard score    | Training time (hh:mm:ss)    |
| ------------------ | --------------------------- | --------------------------- |
| MLP_NET_V1         | 0.45856                     | 00:05:22                    |
| CNN_NET_V1         | 0.51933                     | 00:21:17                    |
| GRU_NET            | 0.00000                     | PyTorch GRU does not work   |
| CNN_NET_V2         | 0.85635                     | 00:01:28                    |
| LSTM_NET           | 0.83425                     | 00:16:16                    |
| MLP_NET_TF         | 0.90055                     | 00:08:20                    |
| CNN_NET_TF         | 0.87845                     | 00:06:18                    |
| GRU_NET_TF         | 0.89502                     | 00:18:55                    |
| LSTM_NET_TF        | 0.88950                     | 00:19:04                    |
| GRU_NET_BIG_TF     | 0.95027                     | 00:22:47                    |
| CONV_LSTM_NET_TF   | 0.93370                     | 00:35:53                    |

The two models GRU_NET_BIG_TF and CONV_LSTM_NET_TF were trained with an extended data set. For this purpose, three new features were added by means of feature engineering. The features are the Fast Fourier Transformation (FFT) of the individual signals. <br> 
In addition, these two models were trained with data created with a sliding window of size 200. All other models were trained with size 100. <br> 

The best model is therefore the GRU_NET_BIG_TF with an accuracy of 95.027%.
