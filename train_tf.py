#---------------------------------------------------------------------------------------------------#
# File name: train_tf.py                                                                            #
# Autor: Chrissi2802                                                                                #
# Created on: 05.10.2022                                                                            #
#---------------------------------------------------------------------------------------------------#
# WISDM - Biometric time series data classification
# Exact description in the functions.
# This file provides the training and evaluation run for TensorFlow.


import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
import numpy as np

import dataset_tf, models_tf, helpers


def train_wisdm_tf():
    """This function performs the training and testing for the WISDM dataset for TensorFlow."""

    # Hyperparameter
    epochs = 2  #500    # For testing 2
    batch_size = 256
    verbose = 1
    
    # Hardware config
    strategy = helpers.hardware_config("GPU")

    # Disable AutoShard
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

    with strategy.scope():

        # Dataset
        CWISDM_Dataset_TF = dataset_tf.WISDM_Dataset_TF(create_new = True, many_features = False, cnn = 0, 
                                                        window_size = 100, step_size = 50)
        train, targets, test = CWISDM_Dataset_TF.get_datasets() 

        path = CWISDM_Dataset_TF.get_path()
        path_models = path.rstrip("Dataset/") + "/Models/"
        acitvity_names = CWISDM_Dataset_TF.get_acitvity_names()

        # Crossvalidation
        k_fold = KFold(n_splits = 5, shuffle = True, random_state = 28)    # For testing n_splits = 2
        val_acc_last = []

        # Numpy array for the predictions
        test_predictions = np.empty([test.shape[0], 6, k_fold.n_splits])

        # Perform the crossvalidation
        for fold, (train_index, test_index) in enumerate(k_fold.split(train, targets)):

            print("Fold:", fold)

            # Data for this fold
            x_train, x_valid = train[train_index], train[test_index]
            y_train, y_valid = targets[train_index], targets[test_index]

            # Wrap data in Dataset objects
            train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).with_options(options)
            valid_data = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(batch_size).with_options(options)
            test_data = tf.data.Dataset.from_tensor_slices((test)).batch(batch_size).with_options(options)

            # Model, choose one
            #model = models_tf.mlp_net_tf(train)
            #model = models_tf.cnn_net_tf(train)    # dataset_tf.WISDM_Dataset_TF(cnn = 1)
            model = models_tf.gru_net_tf(train)
            #model = models_tf.lstm_net_tf(train)
            #model = models_tf.gru_net_big_tf(train)    # dataset_tf.WISDM_Dataset_TF(many_features = True, 
                                                                                    # window_size = 200, step_size = 100)
            #model = models_tf.conv_lstm_net_tf(train)  # dataset_tf.WISDM_Dataset_TF(many_features = True, cnn = 2, 
                                                                                    # window_size = 200, step_size = 100)
            print(model.summary())

            model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

            learning_rate = ReduceLROnPlateau(monitor = "val_loss", factor = 0.5, patience = 10, verbose = verbose)
            early_stopping = EarlyStopping(monitor = "val_loss", patience = 50, verbose = verbose, mode = "min", 
                                           restore_best_weights = True)
            model_checkpoint = ModelCheckpoint(path_models + model.name + str(fold) + ".hdf5", monitor = "val_loss", 
                                               verbose = verbose, save_best_only = True, mode = "auto", save_freq = "epoch")

            # Training
            history = model.fit(train_data, 
                                validation_data = valid_data, 
                                epochs = epochs,
                                verbose = 2,    # for debugging verbose
                                batch_size = batch_size, 
                                callbacks = [learning_rate, early_stopping, model_checkpoint])

            # Plot training and testing curves
            helpers.plot_loss_and_acc(len(history.history["loss"]), history.history["loss"], 
                                      [acc * 100.0 for acc in history.history["accuracy"]], 
                                      history.history["val_loss"], 
                                      [acc * 100.0 for acc in history.history["val_accuracy"]], str(fold))
            
            val_acc_last.append(np.around(100.0 * history.history["val_accuracy"][-1], 2))    # safe last validation accuracy

            # Plot confusion matrix
            valid_predictions = np.argmax(model.predict(valid_data, batch_size = batch_size), axis = 1)
            helpers.plot_conf_matrix(valid_predictions, y_valid, acitvity_names, fold)                                          
            
            # Save predictions 
            test_predictions[:, :, fold] = model.predict(test_data, batch_size = batch_size)

            print()

        # Save submissions
        CWISDM_Dataset_TF.write_submissions_max(test_predictions)
        print("Last validation accuracy for every fold:", val_acc_last)
        print("Training, validation and testing completed!")


if (__name__ == "__main__"):
    
    Pr = helpers.Program_runtime()   # Calculate program runtime 

    train_wisdm_tf()

    Pr.finish()
    
