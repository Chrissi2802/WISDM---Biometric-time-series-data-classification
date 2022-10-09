#---------------------------------------------------------------------------------------------------#
# File name: helpers.py                                                                             #
# Autor: Chrissi2802                                                                                #
# Created on: 05.08.2022                                                                            #
#---------------------------------------------------------------------------------------------------#
# WISDM - Biometric time series data classification
# Exact description in the functions.
# This file provides auxiliary classes and functions for neural networks.


import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix


def plot_loss_and_acc(epochs, train_losses, train_acc, test_losses = [], test_acc = [], fold = -1):
    """This function plots the loss and accuracy for training and, if available, for validation."""
    # Input:
    # epochs; integer, Number of epochs
    # train_losses; list, Loss during training for each epoch
    # train_acc; list, Accuracy during training for each epoch
    # test_losses; list default [], Loss during validation for each epoch
    # test_acc; list default [], Accuracy during validation for each epoch
    # fold; integer default -1, Cross-validation run

    fig, ax1 = plt.subplots()
    xaxis = list(range(1, epochs + 1))

    # Training
    # Loss
    trl = ax1.plot(xaxis, train_losses, label = "Training Loss", color = "red")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    # Accuracy
    ax2 = ax1.twinx()
    tra = ax2.plot(xaxis, train_acc, label = "Training Accuracy", color = "fuchsia")
    ax2.set_ylabel("Accuracy in %")
    ax2.set_ylim(0.0, 100.0)
    lns = trl + tra # Labels

    # Test
    if ((test_losses != []) and (test_acc != [])):
        # Loss
        tel = ax1.plot(xaxis, test_losses, label = "Validation Loss", color = "lime")

        # Accuracy
        tea = ax2.plot(xaxis, test_acc, label = "Validation Accuracy", color = "blue")

        lns = trl + tel + tra + tea    # Labels

    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs)

    if (fold == -1):
        fold1 = ""
        fold2 = ""
    else:
        fold1 = " Fold " + str(fold)
        fold2 = "_Fold_" + str(fold)
        
    plt.title("Loss and Accuracy" + fold1)
    fig.savefig("Loss_and_Accuracy" + fold2 + ".png")
    plt.show()


def count_parameters_of_model(model):
    """This function counts all parameters of a passed PyTorch model."""
    # Input:
    # model; the pytorch model
    # Output:
    # params; integer, Number of parameters

    params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    return params


class Program_runtime():
    """Class for calculating the programme runtime and outputting it to the console."""

    def __init__(self):
        """Initialisation of the class (constructor). Automatically saves the start time."""

        self.begin()

    def begin(self):
        """This method saves the start time."""

        self.__start = datetime.now()   # start time

    def finish(self, print = True):
        """This method saves the end time and calculates the runtime."""
        # Input:
        # print; boolean, default false, the start time, end time and the runtime should be output to the console
        # Output:
        # self.__runtime; integer, returns the runtime

        self.__end = datetime.now() # end time
        self.__runtime = self.__end - self.__start  # runtime

        if (print == True):
            self.show()

        return self.__runtime

    def show(self):
        """This method outputs start time, end time and the runtime on the console."""

        print()
        print("Start:", self.__start.strftime("%Y-%m-%d %H:%M:%S"))
        print("End:  ", self.__end.strftime("%Y-%m-%d %H:%M:%S"))
        print("Program runtime:", str(self.__runtime).split(".")[0])    # Cut off milliseconds
        print()


def hardware_config(device = "GPU"):
    """This function configures the hardware."""
    # Input:
    # device; string default GPU, which device to use, TPU or GPU
    # Output:
    # strategy; tensorflow MirroredStrategy

    if (device == "TPU"):
        # TPU, use only if TPU is available
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
    else:
        # GPU, if not available, CPU is automatically selected
        gpus = tf.config.list_logical_devices("GPU")
        strategy = tf.distribute.MirroredStrategy(gpus)

    return strategy


def plot_conf_matrix(valid_predictions, y_valid, acitvity_names, fold):
    """This function plots the confusion matrix for the validation data."""
    # Input:
    # valid_predictions; NumPy array, Array of predictions
    # y_valid; NumPy array, Array of the true labels
    # acitvity_names; list, list of activity names
    # fold; integer, Cross-validation run

    conf_matrix = confusion_matrix(y_valid, valid_predictions)
    plot_confusion_matrix(conf_mat = conf_matrix, class_names = acitvity_names, show_normed = True, figsize = (10, 7), colorbar = True)
    plt.title("Confusion matrix Fold " + str(fold))
    plt.savefig("Confusion_matrix_Fold_" + str(fold) + ".png")   
    plt.show()


if (__name__ == "__main__"):
    
    # calculating the programme runtime
    Pr = Program_runtime()
    # Code here
    Pr.finish(print = True)

    # configures the hardware
    strategy = hardware_config("GPU")

    with strategy.scope():
        pass
        # Code here
    
