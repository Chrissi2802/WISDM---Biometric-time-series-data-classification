#---------------------------------------------------------------------------------------------------#
# File name: train.py                                                                               #
# Autor: Chrissi2802                                                                                #
# Created on: 14.08.2022                                                                            #
#---------------------------------------------------------------------------------------------------#
# WISDM - Biometric time series data classification
# Exact description in the functions.


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

import datasets, models, helpers


#------------------------------------------------------------------------------------------------------------------#
#                                                MLP & CNN 1D & GRU                                                #
#------------------------------------------------------------------------------------------------------------------#


def train_mlp_cnnv1_gru(model, epochs, batch_size, learning_rate, cuda, plots):
    """This function trains a model (MLP, CNN 1D, GRU) for classification using the WISDM dataset."""
    # Input: 
    # model; pytorch model
    # epochs; number of epochs
    # batch_size; number training batch size
    # learning_rate; number learning rate
    # cuda; boolean train the model on cuda or not
    # plots; boolean produce plots of train and test losses and accuracies
    # Output:
    # model; the pytorch trained model
    # train_losses; where train losses are a simple python list

    # Load the data and put it into the DataLoader
    print("Prepare the data for training ...")
    dataset_train = datasets.WISDM_Dataset("train")

    if (model.__class__.__name__ == "GRU_NET"):
        sw = True   # use sliding_window
    else:
        sw = False

    dl_train = dataset_train.dataloading(batch_size, True, True, sliding_window = sw)
    print("Preparation of the data completed!")

    if ((plots == True) and (sw == False)):
        dataset_train.visualisation()

    train_losses = []
    train_acc = []

    if (cuda == True):
        # moving model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(device)   # only for testing
        model = model.to(device)
    else:
        device = "cpu"

    loss = nn.CrossEntropyLoss()    # Classification => Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    # train Model
    print("Start of training ...")
    for epoch in range(epochs):
        
        # Training
        model.train()
        running_loss = 0.0
        train_accuracy = 0.0

        for batch in dl_train:
            x_batch, y_batch = batch[:, 2:5], batch[:, 1].long()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()   # reset gradients to avoid incorrect calculation
            prediction = model.forward(x_batch)
            l = loss(prediction, y_batch)
            l.backward()
            optimizer.step()
            running_loss += l.item()

            # Accuracy
            top_p, top_class = torch.exp(prediction).topk(1, dim = 1)
            equals = top_class == y_batch.view(*top_class.shape)
            train_accuracy += torch.mean(equals.type(torch.FloatTensor))

        # Save loss and accuracy for training
        train_losses.append(running_loss / len(dl_train))
        train_acc.append((train_accuracy / len(dl_train)) * 100)    

        # Output current status on console
        print("Epoch: {:03d}/{:03d}".format(epoch + 1, epochs),
              "Training loss: {:.3f}".format(running_loss / len(dl_train)),
              "Training Accuracy: {:.3f}".format((train_accuracy / len(dl_train)) * 100))

    print("Training completed!")

    # ploting
    if (plots == True):
        helpers.plot_loss_and_acc(epochs, train_losses, train_acc)

    return model, train_losses


def evaluation(model, cuda):
    """This function performs the evaluation of the model with the test data set."""
    # Input:
    # model; the pytorch trained model

    # Load the data and put it into the DataLoader
    print("Prepare the data for validation ...")
    dataset_test = datasets.WISDM_Dataset("test")
    dl_test = dataset_test.dataloading(1, False, False)
    print("Preparation of the data completed!")

    if (cuda == True):
        # moving model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(device)   # only for testing
        model = model.to(device)
    else:
        device = "cpu"

    # monitoring - evaluate test loss
    print("Start of validation ...")
    with torch.no_grad():   # no gradients, because just monitoring, no optimization

        model.eval()    # Set the model to evaluation mode
        old_id = 0
        bestpred = []
        model_name = model.__class__.__name__

        for batch in dl_test:

            new_id = int(batch[0, 0].item())
            x_batch, y_batch = batch[:, 2:5], batch[:, 1].long()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            prediction = model(x_batch)
            top_p, top_class = torch.exp(prediction).topk(1, dim = 1)
            top_class = top_class.long()

            # Check sample_id and write value in txt if changed
            if (old_id == new_id):
                if (bestpred == []):
                    bestpred = top_class
                else:
                    bestpred = torch.cat((bestpred, top_class))
            else:
                dataset_test.writepredictions(old_id, bestpred, model_name) # Safe predictions
                bestpred = top_class

            old_id = new_id

    dataset_test.writepredictions(old_id, bestpred, model_name) # Safe last prediction

    print("Validation completed!")


def testdifhyperparameter():
    """This funciton trys different values of the hyper-parameter (user parameters) settings."""

    base_model_list = [models.MLP_NET_V1(), models.CNN_NET_V1(), models.GRU_NET(3, 4, 2, 6)]
    batch_size = [16, 64, 128, 256, 512]    # Batch size
    learning_rate = [0.01, 0.001, 0.0001]   # Learning rate

    # Test different models, batch sizes and learning rates
    for base_model in base_model_list:  # different models
        for ba in batch_size:           # different batch sizes
            for lr in learning_rate:    # different learning rates
                print("Model:", base_model.__class__.__name__, " |  Optimizer: Adam  |  Batch size:", ba, " |  Learning rate:", lr)
                model, losses = train_mlp_cnnv1_gru(base_model, 50, ba, lr, True, True)
                print(model)
                evaluation(model)
                print()


def run_train_mlp_cnnv1_gru():
    """This function performs the training and validation for the MLPm CNN V1 and GRU."""
    # The model, hyperparameters and other settings can be changed directly in this function.

    epochs = 50             # number of epochs
    batch_size = 256        # training batch size
    learning_rate = 0.001   # learning rate
    cuda = True             # true or false to train the model on cuda or not
    plots = True            # true or false to produce plots of train losses and accuracies

    mlp_v1_model = models.MLP_NET_V1()
    cnn_v1_model = models.CNN_NET_V1()
    gru_model = models.GRU_NET(3, 4, 2, 6)

    # train the model
    model, train_losses = train_mlp_cnnv1_gru(mlp_v1_model, epochs, batch_size, learning_rate, cuda, plots)
    print(model)

    # only for testing
    #print(train_losses)
    print("Parameters of the model:", helpers.count_parameters_of_model(model))  
    torch.save(model, "model.pth")
    #model = torch.load("model.pth")
    #testdifhyperparameter()

    # evaluate the model
    evaluation(model, cuda)

#------------------------------------------------------------------------------------------------------------------#
#                                                  CNN 2D & LSTM                                                   #
#------------------------------------------------------------------------------------------------------------------#


def plot_histogram(data, name):
    _, ax = plt.subplots(figsize=(10, 3))
    ax.hist(data, bins=100, range=(-4,4))
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title(f"Histogram {name}")
    plt.show()


def train_cnnv2_lstm(model, dl_train, model_type = "RNN", learning_rate = 0.1):
    """This function trains a model (CNN 2D, LSTM) for classification using the WISDM dataset."""

    # set the device which will be used to train the model
    device = torch.device('cuda:0' if torch.cuda.device_count() >= 1 else 'cpu')
    model = model.to(device)

    # use CrossEntropyLoss for classification problem
    loss = nn.CrossEntropyLoss()
    # use SGD optimization
    optimizer = optim.SGD(model.parameters(), lr = learning_rate)
    train_losses = []
    train_accuracies = []
    best_loss = 10000.
    best_model = model
    lr_idx = 0
    epoch = 0

    while (learning_rate > 1e-6):
        epoch += 1
        # set the model in training mode      
        model.train()
        train_loss, train_acc = 0., 0.

        for batch in dl_train:            
            # send the input to the device
            x_batch, y_batch = batch[0].to(device), batch[1].long().to(device)

            if (model_type == 'CNN'):
                x_batch = x_batch.unsqueeze(1) # change size to [num_batch, channel, height, width]

            # perform a forward pass and calculate the training loss
            predictions = model(x_batch)
            l = loss(predictions, y_batch)

            # zero out the gradients, perform the backpropagation step, and update the weights
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_loss += l.item()*len(x_batch)
            train_acc += (predictions.argmax(dim=1) == y_batch).type(torch.float).sum().item()

        train_loss /= len(dl_train.dataset)
        train_acc /= len(dl_train.dataset)/100.
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
                  
        print(f"Epoch {'{:03d}'.format(epoch)} - Training loss: {'{:.3f}'.format(train_loss)} - Training accuracy: {'{:.2f}'.format(train_acc)}% - Learning rate: {learning_rate}")

        # save the best model
        if (best_loss > train_loss):
            best_loss = train_loss
            best_model = model
            lr_idx = epoch
        
        # reduce the learning rate, if the loss has not reduced in the past epochs
        if (lr_idx + 3 <= epoch):
            learning_rate /= 2.
            optimizer.param_groups[0]['lr'] = learning_rate
            lr_idx = epoch
            model = best_model
            
    # plot training loss and accuracy
    helpers.plot_loss_and_acc(epoch, train_losses, train_accuracies)
    
    return best_model


def ouput(model, time_length, batch_size, inverse_mapping_labels, model_type = "RNN"):

    # set the device which will be used to train the model
    device = torch.device('cuda:0' if torch.cuda.device_count() >= 1 else 'cpu')
    model = model.to(device)

    # Test data
    dataset_test = datasets.WISDM_Dataset("test")
    normalized_data_test = pd.DataFrame(dataset_test.data_tensor)
    normalized_data_test.set_axis(["test-id", "subjects", "x", "y", "z"], axis = "columns", inplace = True)

    output = open("./Predictions/result.csv", "w")
    output.write('sample_id,activity\n')
    model.eval()

    for i in range(259):

        data_test = normalized_data_test.loc[normalized_data_test['test-id'] == i]
        dataset = datasets.Create_Dataset(data_test[['x', 'y', 'z']], data_test[['test-id','subjects']], time_length, sliding_step=time_length)
        dl_test = DataLoader(dataset, batch_size, shuffle=False)
        y_hat = []

        for batch in dl_test:
            if (model_type == "RNN"):
                y_hat.append(model(batch[0].to(device)))
            else:
                # Shape: [samples, channel=1, height, width]
                y_hat.append(model(batch[0].unsqueeze(1).to(device)))

        y_hat = torch.cat(y_hat, dim=0)
        y_hat = y_hat.argmax(dim=1)
        output.write(f"{i},{inverse_mapping_labels[y_hat.bincount().argmax().item()]}"+"\n")

    output.close()


def run_train_cnnv2_lstm():
    """This function performs the training and validation for the CNN V2 and LSTM."""
    # The model, hyperparameters and other settings can be changed directly in this function.

    # Data preprocessing
    dataset_train = datasets.WISDM_Dataset("train")
    normalized_data = pd.DataFrame(dataset_train.data_tensor)
    normalized_data.set_axis(["subjects", "labels", "x", "y", "z"], axis = "columns", inplace = True)
    inverse_mapping_labels = dataset_train.activity_dic_inv

    # Plot histogram
    plot_histogram(normalized_data['x'], 'x')
    plot_histogram(normalized_data['y'], 'y')
    plot_histogram(normalized_data['z'], 'z')
    dataset_train.visualisation()

    # define time_length, sliding_step and batch_size
    time_length = 128
    sliding_step = 64
    batch_size = 32
    dataset = datasets.Create_Dataset(normalized_data[['x', 'y', 'z']], normalized_data[['subjects','labels']], time_length, sliding_step)
    dl_train = DataLoader(dataset, batch_size, shuffle=True)

    rnn_model = models.LSTM_NET(input_dim = 3, hidden_dim = 32, time_length = time_length)
    print(rnn_model)
    print("Parameters of the model:", helpers.count_parameters_of_model(rnn_model))  

    rnn_model = train_cnnv2_lstm(rnn_model, dl_train, model_type='RNN', learning_rate = 0.1)
    torch.save(rnn_model, "rnn_model.pth")

    #cnn_model = models.CNN_NET_V2(height = time_length, width = 3)
    #print(cnn_model)
    #print("Parameters of the model:", helpers.count_parameters_of_model(cnn_model))  

    #cnn_model = train_cnnv2_lstm(cnn_model, dl_train, model_type='CNN', learning_rate = 0.1)
    #torch.save(cnn_model, "cnn_model.pth")

    # testing
    ouput(rnn_model, time_length, batch_size, inverse_mapping_labels, model_type = "RNN")
    #ouput(cnn_model, time_length, batch_size, inverse_mapping_labels, model_type = "CNN")


if (__name__ == "__main__"):
    
    Pr = helpers.Program_runtime()   # Calculate program runtime 
    
    run_train_mlp_cnnv1_gru()

    #run_train_cnnv2_lstm()

    Pr.finish()
    
