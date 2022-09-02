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


def plot_loss_and_acc(epochs, train_losses, train_acc, test_losses = [], test_acc = []):
    """This function plots the loss and accuracy for training and, if available, for validation."""
    # Input:
    # epochs; integer, Number of epochs
    # train_losses; list, Loss during training for each epoch
    # train_acc; list, Accuracy during training for each epoch
    # test_losses; list default [], Loss during validation for each epoch
    # test_acc; list default [], Accuracy during validation for each epoch

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
    plt.title("Loss and Accuracy")
    fig.savefig("Loss_and_Accuracy.png")
    plt.show()


def count_parameters_of_model(model):
    """This function counts all parameters of a passed model."""
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


if (__name__ == "__main__"):
    pass
    
