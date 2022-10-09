#---------------------------------------------------------------------------------------------------#
# File name: dataset_tf.py                                                                          #
# Autor: Chrissi2802                                                                                #
# Created on: 03.10.2022                                                                            #
#---------------------------------------------------------------------------------------------------#
# WISDM - Biometric time series data classification
# Exact description in the functions.
# This file provides the dataset for tensorflow.


import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import seaborn as sns
import warnings
from scipy import stats


class WISDM_Dataset_TF():
    """Class to design a WISDM Dataset for TensorFlow."""

    def __init__(self, create_new = True, many_features = False, cnn = 0, window_size = 100, step_size = 50):
        """Initialisation of the class (constructor). It prepares the data to be used for training and testing."""
        # Input:
        # create_new; boolean default True, create new data or load old data
        # many_features; boolean default False, many features should be added
        # cnn; integer default 0, If a CNN is used, the data must be reshaped
        # window_size; integer, size of the sliding window
        # step_size; integer, shifting of the sliding window

        print("Prepare the dataset for training and testing ...")

        self.create_new = create_new
        self.many_features = many_features
        self.cnn = cnn
        self.window_size = window_size
        self.step_size = step_size
        self.path = "./Datasets/" 
        self.path_plots = self.path.rstrip("Datasets/") + "/Plots/"
        self.column_names = ["ID", "Activity", "X", "Y", "Z"]
        self.acitvity_names = ["Walking", "Jogging", "Climbing stairs", "Sitting", "Standing", "Kicking soccer ball"]
        self.activity_dic = {"A": 0, "B": 1, "C": 2, 
                             "D": 3, "E": 4, "M": 5}
        # Inverted dictionary for reconversion
        self.activity_dic_inv = {item: element for element, item in self.activity_dic.items()}

        if (self.create_new == True):
            self.filelist_train = [txt for txt in os.listdir(self.path + "train/") if txt[-4:] == ".txt"]
            self.__create_df()
            self.__visualisation()
            self.__normalize()
            self.__feature_engineering()

            if (self.cnn >= 1):
                self.__reshape()

                if (self.cnn == 2):
                    self.__reshape_complex()
        else:
            self.load_dataset_numpy()

        print("Preparation of the data completed!")

    def __create_df(self):
        """This method creates Panda's DataFrames and reads the data into them."""

        # Training data
        self.dataset_train = pd.DataFrame()

        for txt in self.filelist_train:
            dataset_temp = pd.read_csv(self.path + "train/" + txt, header = None, names = self.column_names, comment = ";")
            self.dataset_train = pd.concat([self.dataset_train, dataset_temp])

        # Test data
        self.dataset_test = pd.read_csv(self.path + "test/test_data_accel_watch.txt", header = None, names = self.column_names, comment = ";")

    def __sliding_window(self, dataset):  
        """This method executes the sliding window."""
        # Input:
        # dataset; DataFrame
        # Output:
        # dataset; DataFrame

        x_list, y_list, z_list = [], [], []
        x_fft_list, y_fft_list, z_fft_list = [], [], []
        self.targets_list = []
        self.test_id_list = []

        # creating overlaping windows of size window-size
        for i in range(0, dataset.shape[0] - self.window_size, self.step_size):

            x_list.append(dataset["X"].values[i: i + self.window_size])
            y_list.append(dataset["Y"].values[i: i + self.window_size])
            z_list.append(dataset["Z"].values[i: i + self.window_size])
            self.targets_list.append(stats.mode(dataset["Activity"][i: i + self.window_size])[0][0])
            
            if (self.many_features == True):
                # Runs an FFT and adds the features
                x_fft_list.append(np.fft.fft(dataset["X"].values[i: i + self.window_size]))
                y_fft_list.append(np.fft.fft(dataset["Y"].values[i: i + self.window_size]))
                z_fft_list.append(np.fft.fft(dataset["Z"].values[i: i + self.window_size]))

            # If label -1, then it is test data where no labels are present. 
            # The ID must be saved there as well. Save the ID that occurs most often.
            if (dataset["Activity"].iloc[0] == -1):
                self.test_id_list.append(stats.mode(dataset["ID"][i: i + self.window_size])[0][0])

        feature_list = [x_list, y_list, z_list]

        if (self.many_features == True):
            feature_list.append(x_fft_list)
            feature_list.append(y_fft_list)
            feature_list.append(z_fft_list)

        feature_list = [np.array(data_list) for data_list in feature_list]
        dataset = np.stack(feature_list, axis = 2)

        return dataset

    def __feature_engineering(self):
        """This method extends and changes the features."""

        # First: train data
        self.dataset_train = self.__sliding_window(self.dataset_train)
        self.targets = np.array(self.targets_list)

        # Second: test data
        self.dataset_test = self.__sliding_window(self.dataset_test)

    def __normalize(self):
        """This method scales / normalises the features."""

        self.dataset_train["Activity"] = self.dataset_train["Activity"].map(self.activity_dic)
        labels = self.dataset_train["Activity"].to_numpy()
        test_id = self.dataset_test["ID"].to_numpy()

        scaler = RobustScaler()

        self.dataset_train = scaler.fit_transform(self.dataset_train[["X", "Y", "Z"]])
        self.dataset_test = scaler.transform(self.dataset_test[["X", "Y", "Z"]])

        self.dataset_train = pd.DataFrame(data = self.dataset_train, columns = ["X", "Y", "Z"])
        self.dataset_train["Activity"] = labels
        self.dataset_test = pd.DataFrame(data = self.dataset_test, columns = ["X", "Y", "Z"])
        self.dataset_test["Activity"] = np.zeros(shape = self.dataset_test["X"].shape) - 1
        self.dataset_test["ID"] = test_id

    def __reshape(self):
        """This method reshapes the data for CNN models."""

        self.dataset_train = np.expand_dims(self.dataset_train, axis = 3)
        self.dataset_test = np.expand_dims(self.dataset_test, axis = 3)
        
    def __reshape_complex(self):
        """This method reshapes the data for complex CNN models."""

        shape_train = self.dataset_train.shape
        shape_test = self.dataset_test.shape
        subsequences = 4    # split the sliding window into four parts

        self.dataset_train = self.dataset_train.reshape((shape_train[0], subsequences, int(shape_train[1] / subsequences),
                                                         shape_train[2], 1))

        self.dataset_test = self.dataset_test.reshape((shape_test[0], subsequences, int(shape_test[1] / subsequences), 
                                                       shape_test[2], 1))

    def get_datasets(self):
        """This method returns the training data, labels and test data."""
        # Output:
        # self.dataset_train, self.targets, self.dataset_test; numpy arrays

        return self.dataset_train, self.targets, self.dataset_test 

    def get_path(self):
        """This method returns the path."""
        # Output:
        # self.path; string

        return self.path

    def get_acitvity_names(self):
        """This method returns the acitvity names."""
        # Output:
        # self.acitvity_names; list

        return self.acitvity_names

    def save_datasets_numpy(self):
        """This method saves the data arrays to a binary file in NumPy .npy format."""
        
        np.save(self.path + "train.npy", self.dataset_train)
        np.save(self.path + "targets.npy", self.targets)
        np.save(self.path + "test.npy", self.dataset_test)
        print("Data saved as NumPy files!")

    def load_dataset_numpy(self):
        """This method loads arrays from .npy files."""

        self.dataset_train = np.load(self.path + "train.npy")
        self.targets = np.load(self.path + "targets.npy")
        self.dataset_test = np.load(self.path + "test.npy")
        print("Data loaded from NumPy files!")

    def write_submissions_max(self, test_predictions):
        """This method writes the predictions from the cross-validation (the maximum of all predictions) into a csv file."""
        # Input:
        # test_predictions; numpy array

        # Every single fold is used. The folds are added, then the same ids are added and the maximum is determined.
        predictions = np.sum(test_predictions, axis = 2)    # Adding the Folds
        predictions = pd.DataFrame(predictions)
        predictions["ID"] = self.test_id_list
        
        predictions = predictions.groupby(["ID"]).sum()     # Summing up classes with the same id
        predictions = predictions.sort_index()
        predictions = predictions.idxmax(axis = 1)          # find highest value and return index

        self.submission = pd.DataFrame({"sample_id": predictions.index, "activity": predictions.values})
        self.submission["activity"] = self.submission["activity"].map(self.activity_dic_inv)
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.submission.to_csv(self.path + now + "_max_submission.csv", index = False)

    def __visualisation(self):
        """This method visualises the data."""

        self.__vis_data_points_per_category()   # Number of data points in each category as bar chart
        self.__vis_category_per_user()          # shows the activities of each category for each user / ID
        self.__vis_sample_series_per_category() # Sample data series for all six categories
        self.__vis_distribution_per_signal()    # shows the distribution of signals by activity
        
    def __vis_data_points_per_category(self):
        """This method displays the number of data points in each category as a bar chart."""

        activity_counts = self.dataset_train["Activity"].value_counts().sort_index()
        mean = activity_counts.mean()

        plt.rcParams["figure.figsize"] = (12, 7)
        plt.bar(self.acitvity_names, activity_counts.values, label = "Number of data points")
        plt.axhline(mean, label = "Mean", color = "red")
        plt.title("Number of datapoints by Activities")
        plt.legend()
        plt.savefig(self.path_plots + "Number_of_datapoints_by_Activities.png")
        plt.show()

    def __vis_category_per_user(self):
        """This method shows the activities of each category for each user / ID."""

        plt.figure(figsize = (18, 6))
        chart = sns.countplot(x = "ID", hue = "Activity", data = self.dataset_train)
        chart.set_xticklabels(chart.get_xticklabels(), rotation = 45)
        chart.set_xlabel("User / ID")
        chart.set_ylabel("Number of data points")
        plt.title("Activities by Users")
        plt.legend()
        plt.savefig(self.path_plots + "Activities_by_Users.png")
        plt.show()

    def __vis_sample_series_per_category(self):
        """This method visualises sample data series for all six categories."""
        
        length = 200
        labels = ["x-signal", "y-signal", "z-signal"]
        x_values = np.linspace(0.0, length * 0.05, length)

        fig, axes = plt.subplots(3, 2, sharex = True, figsize = (18, 9))

        for i in range(6):  # Fill all subplots
            start = i * 2400
            xyz = self.dataset_train.iloc[start:start + length, 2:5]   # data for this plot

            if (i < 3):
                row = i
                col = 0
            else:
                row = i - 3
                col = 1
                
            axes[row, col].plot(x_values, xyz.values)
            axes[row, col].set_title(self.acitvity_names[i])
            axes[row, col].grid()

        plt.rcParams["figure.figsize"] = (12, 7)
        fig.legend(labels)
        plt.setp(axes[-1, :], xlabel = "Time [s]")
        plt.suptitle("Sample data series of each category")
        plt.savefig(self.path_plots + "Sample_data_series_of_each_category.png")
        plt.show()

    def __vis_distribution_per_signal(self):
        """This method shows the distribution of signals by activity."""

        warnings.filterwarnings("ignore")

        for axis in ["X", "Y", "Z"]:
            sns.FacetGrid(self.dataset_train, hue = "Activity", size = 6).map(sns.distplot, axis).add_legend()
            plt.suptitle("Distribution of signal " + axis + " by activity")
            plt.savefig(self.path_plots + "Distribution_of_signal_" + axis + "_by_activity.png")
            plt.show()

        warnings.filterwarnings("default")


if (__name__ == "__main__"):
    
    CWISDM_Dataset_TF = WISDM_Dataset_TF(create_new = True, many_features = False, cnn = 0, 
                                         window_size = 100, step_size = 50)
    train, targets, test = CWISDM_Dataset_TF.get_datasets()        

    print(train.shape, targets.shape, test.shape)
    #CWISDM_Dataset_TF.save_datasets_numpy()
    #CWISDM_Dataset_TF.write_submissions_max(np.ones((8701, 6, 5)))

