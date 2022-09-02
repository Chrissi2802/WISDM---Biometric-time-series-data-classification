#---------------------------------------------------------------------------------------------------#
# File name: datasets.py                                                                            #
# Autor: Chrissi2802                                                                                #
# Created on: 14.07.2022                                                                            #
#---------------------------------------------------------------------------------------------------#
# WISDM - Biometric time series data classification
# Exact description in the functions.
# This file provides the dataset.


import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


class WISDM_Dataset():
    """Class to design a WISDM Dataset."""

    def __init__(self, mode):
        """Initialisation of the class (constructor). It prepares the data to be used for training and validation."""
        # Input:
        # mode; string, train or test data

        # Load the data
        self.column_names = ["ID", "Activity", "X", "Y", "Z"]
        self.acitvity_names = ["Walking", "Jogging", "Climbing stairs", "Sitting", "Standing", "Kicking soccer ball"]
        self.activity_dic = {"A": 0, "B": 1, "C": 2, 
                             "D": 3, "E": 4, "M": 5}

        # Inverted dictionary for reconversion
        self.activity_dic_inv = {item: element for element, item in self.activity_dic.items()}

        self.folder = "./Datasets/" + mode + "/"
        self.filelist = [txt for txt in os.listdir(self.folder) if txt[-4:] == ".txt"]
        self.data_tensor = []
        self.data_tensor_raw = []

        self.__create_tensor()

        self.predfname = None
        
    def __create_tensor(self):
        """This method combines all text files into one big tensor."""

        for txt in self.filelist:
            self.data = pd.read_csv(self.folder + txt, header = None, names = self.column_names, comment = ";") # load data

            # Replaced the activity description currently with letters with numbers
            self.data["Activity"] = self.data["Activity"].map(self.activity_dic)

            # safe the raw data in a tensor
            if (self.data_tensor_raw == []):
                self.data_tensor_raw = torch.tensor(self.data.values).float()
            else:
                self.data_tensor_raw = torch.cat((self.data_tensor_raw, torch.tensor(self.data.values).float()))

            self.__normalize_feature()  # normalizes all features

            # safe the normalized data in a tensor
            if (self.data_tensor == []):
                self.data_tensor = torch.tensor(self.data.values).float()
            else:
                self.data_tensor = torch.cat((self.data_tensor, torch.tensor(self.data.values).float()))

    def __normalize_feature(self):
        """This method normalizes all features."""

        for dim in ["X", "Y", "Z"]:
            # normalize the data
            mue = np.mean(self.data[dim])     # Mean
            sigma = np.std(self.data[dim])    # Standard deviation
            self.data[dim] = (self.data[dim] - mue) / sigma

    def dataloading(self, batch_size, shuffle, drop_last, sliding_window = False):
        """This method fills the DataLoader."""
        # Input:
        # batch_size; integer, batch size
        # shuffle; boolean, shuffle the data in the DataLoader
        # drop_last; boolean, delete last batch (Sometimes incomplete)
        # sliding_window; boolean

        if (sliding_window == True):
            self.data_tensor = self.slid_win(self.data_tensor, 100, 50)

        # Data loading
        dl = DataLoader(self.data_tensor, batch_size = batch_size, shuffle = shuffle, drop_last = drop_last)

        return dl

    def writepredictions(self, sample_id, prediction, model_name):
        """This method evaluates the passed predictions and writes a new line into a text file for each sample_id."""
        # Input:
        # sample_id; integer, current sample_id
        # prediction; torch tensor, Contains the predicted labels for a sample_id
        # model_name; string, name of the ANN model

        # Create a unique name for the text file
        if (self.predfname == None):
            self.predfname = model_name + "_Predictions_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
            
            # Save the header
            file = open("./Predictions/" + self.predfname, "a")
            file.write("sample_id,activity\n")
            file.close()
        
        # Determine the activity that was predicted the most 
        # And convert this number back to the corresponding letter
        activity = prediction.mode(dim = 0)[0].item()
        activity = self.activity_dic_inv.get(activity)

        # Save these two values in a text file (sample_id,activity)
        file = open("./Predictions/" + self.predfname, "a")
        file.write(str(sample_id) + "," + activity + "\n")
        file.close()

    def visualisation(self):
        """This method visualises the data."""

        self.__vis_data_points_per_category()   # Number of data points in each category as bar chart
        self.__vis_sample_series_per_category() # Sample data series for all six categories
        
    def __vis_data_points_per_category(self):
        """This method displays the number of data points in each category as a bar chart."""

        activity_counts = torch.unique(self.data_tensor[:, 1].long(), sorted = True, return_counts = True)
        mean = torch.mean(activity_counts[1].float())
        
        plt.rcParams["figure.figsize"] = (12, 7)
        plt.bar(self.acitvity_names, activity_counts[1], label = "Number of data points")
        plt.axhline(mean, label = "Mean", color = "red")
        plt.title("Number of datapoints by Activities")
        plt.legend()
        plt.savefig("Number_of_datapoints_by_Activities.png")
        plt.show()

    def __vis_sample_series_per_category(self):
        """This method visualises sample data series for all six categories."""
        
        length = 200
        labels = ["x-signal", "y-signal", "z-signal"]
        x_values = np.linspace(0.0, length * 0.05, length)

        fig, axes = plt.subplots(3, 2, sharex = True, figsize = (18, 9))

        for i in range(6):  # Fill all subplots
            start = i * 2400
            tensorxyz = self.data_tensor_raw[start:start + length, 2:5]   # data for this plot

            if (i < 3):
                row = i
                col = 0
            else:
                row = i - 3
                col = 1
                
            axes[row, col].plot(x_values, tensorxyz)
            axes[row, col].set_title(self.acitvity_names[i])
            axes[row, col].grid()

        fig.legend(labels)
        plt.setp(axes[-1, :], xlabel = "Time [s]")
        plt.suptitle("Sample data series of each category")
        plt.savefig("Sample_data_series_of_each_category.png")
        plt.show()
    
    def slid_win(self, data, window_size, step_size):
        """This method implements a sliding window."""
        # Input:
        # window_size; integer
        # step_size; integer
        # Output:

        boundary = int(np.floor((len(data) - window_size) / step_size))
        output = []

        for i in range(boundary):
            lower = int(i * step_size)
            upper = lower + window_size
            local_data = data[lower:upper]

            if (output != []):
                output = torch.cat((output, local_data))
            else:
                output = local_data

        return output


class Create_Dataset(Dataset):
    """Class to design a Dataset."""

    def __init__(self, X, Y, time_length, sliding_step):
        """Initialisation of the class (constructor)."""
        # Input:
        # X
        # Y
        # time_length
        # sliding_step
        
        super().__init__()
        
        data = []
        labels = []

        for i in range(0, len(X) - time_length + 1, sliding_step):

            if (Y.values[i].all() == Y.values[i + time_length - 1].all()):
                data.append(torch.from_numpy(X.values[i : i + time_length]))

                if ('labels' in Y):
                    labels.append(Y['labels'].values[i])
                else:
                    labels.append(Y['test-id'].values[i])

        self.data = torch.stack(data) # Shape = [num_samples, time_length, features=3]
        self.labels = torch.tensor(labels) # Shape = [num_samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

