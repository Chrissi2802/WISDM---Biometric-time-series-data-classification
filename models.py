#---------------------------------------------------------------------------------------------------#
# File name: models.py                                                                              #
# Autor: Chrissi2802                                                                                #
# Created on: 14.07.2022                                                                            #
#---------------------------------------------------------------------------------------------------#
# WISDM - Biometric time series data classification
# Exact description in the functions.
# This file provides the models.


import torch
import torch.nn as nn


class MLP_NET_V1(nn.Module):
    """Class to design a MLP model."""

    def __init__(self):
        """Initialisation of the class (constructor)."""

        super().__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim = 1)

        self.bnin = nn.BatchNorm1d(16)
        self.bnbout = nn.BatchNorm1d(32)

        self.linin = nn.Linear(3, 16, bias = True)
        self.linbout = nn.Linear(16, 32, bias = True)
        self.linout = nn.Linear(32, 6, bias = True)

    def forward(self, input_data):
        """The layers are stacked to transport the data through the neural network for the forward part."""
        # Input: 
        # input_data; torch.Tensor
        # Output:
        # x; torch.Tensor

        x = self.linin(torch.flatten(input_data, 1))
        x = self.bnin(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linbout(x)
        x = self.bnbout(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linout(x)
        x = self.softmax(x)

        return x
        

class CNN_NET_V1(nn.Module):
    """Class to design a CNN model."""

    def __init__(self):
        """Initialisation of the class (constructor)."""

        super().__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim = 1)

        self.bncnn1 = nn.BatchNorm1d(64)
        self.bncnn2 = nn.BatchNorm1d(128)
        self.bncnn3 = nn.BatchNorm1d(256)
        self.bnbout = nn.BatchNorm1d(64)

        self.cnn1 = nn.Conv1d(3, 64, 3, padding = 2)
        self.cnn2 = nn.Conv1d(64, 128, 3, padding = 1)
        self.cnn3 = nn.Conv1d(128, 256, 3, padding = 1)

        self.avgpool = nn.AvgPool1d(3)

        self.linbout = nn.Linear(256, 64, bias = True)
        self.linout = nn.Linear(64, 6, bias = True)


    def forward(self, input_data):
        """The layers are stacked to transport the data through the neural network for the forward part."""
        # Input: 
        # input_data; torch.Tensor
        # Output:
        # x; torch.Tensor

        # Input dimension: batch_size, features
        x = input_data.unsqueeze(2) # add one dimension 

        # Input dimension: batch_size, 3, 1
        x = self.cnn1(x)
        x = self.bncnn1(x)       
        x = self.relu(x)

        # Input dimension: batch_size, 64, 3
        x = self.cnn2(x)
        x = self.bncnn2(x)
        x = self.relu(x)

        # Input dimension: batch_size, 128, 3
        x = self.cnn3(x)
        x = self.bncnn3(x)
        x = self.relu(x)

        # Input dimension: batch_size, 256, 3
        x = self.avgpool(x)

        # Input dimension: batch_size, 256, 1
        x = self.linbout(torch.flatten(x, 1))
        x = self.bnbout(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Input dimension: batch_size, 64
        x = self.linout(x)
        x = self.softmax(x)
        # Output dimension: batch_size, 6 

        return x


class CNN_NET_V2(nn.Module):
    """Class to design a CNN model."""

    def __init__(self, height, width):
        """Initialisation of the class (constructor)."""
        # Input:
        # height; integer
        # width; integer

        super().__init__()
        
        dropout = 0.2

        self.net = nn.Sequential(nn.Conv2d(1, 16, kernel_size=2, padding=1),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=2),
                                 nn.Dropout(dropout),
                                 nn.Conv2d(16, 32, kernel_size=2, padding=1),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=2),
                                 nn.Dropout(dropout),
                                 nn.Flatten())
    
        xs = self.net(torch.rand(1, 1, height, width))
        
        self.net2 = nn.Sequential(nn.Linear(xs.shape[1], 128),
                                  nn.ReLU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(128, 6))

    def forward(self, input_data):
        """The layers are stacked to transport the data through the neural network for the forward part."""
        # Input: 
        # input_data; torch.Tensor
        # Output:
        # x; torch.Tensor

        x = self.net(input_data)
        x = self.net2(x)

        return x


class LSTM_NET(nn.Module):
    """Class to design a LSTM model."""
        
    def __init__(self, input_dim, hidden_dim, time_length):
        """Initialisation of the class (constructor)."""
        # Input:
        # input_dim, integer 
        # hidden_dim; integer
        # time_length; integer

        super().__init__()        

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first = True)
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.Dropout(0.2),
                                 nn.Linear(time_length * hidden_dim, 128),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(128, 6))    

    def forward(self, input_data):
        """The layers are stacked to transport the data through the neural network for the forward part."""
        # Input: 
        # input_data; torch.Tensor
        # Output:
        # x; torch.Tensor

        x, h = self.lstm(input_data)
        x = self.net(x)

        return x


class GRU_NET(nn.Module):
    """Class to design a GRU model."""

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """Initialisation of the class (constructor)."""
        # Input:
        # input_size
        # sliding_window size; integer
        # hidden_size; integer
        # num_layers; integer
        # output_size; integer

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first = True)
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.Linear(self.hidden_size, self.output_size, bias = True))

    def forward(self, input_data):
        """The layers are stacked to transport the data through the neural network for the forward part."""
        # Input: 
        # input_data; torch.Tensor
        # Output:
        # x; torch.Tensor
        # h; torch.Tensor

        x, h = self.gru(input_data)
        x = self.net(x)

        return x

        