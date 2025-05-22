import pandas as pd
import numpy as np
import os

def read():
    '''
    Function that reads through all the xlsx files in the same directory and determines the best hyperparameters for training of neural network
    
    Parameters:
    None, but xlsx files in same directory is required

    Returns:
    chosen_lam (float or int): The chosen regularization hyperparameter for model
    chosen_nnode (int): The chosen number of nodes per hidden layer for model
    chosen_layers (int): The chosen number of hidden layers for model
    chosen_reg (string): The chosen regularization type ('Lasso' or 'Ridge')
    chosen_epoch (int): The chosen maximum epoch

    '''
    list_layers = []
    list_reg = []
    list_lambda = []
    list_nnode = []
    list_epoch_of_min = []
    list_tr_loss = []
    list_val_loss = []

    for file in os.listdir(os.curdir):
        if file.endswith(".xlsx"):
            df = pd.read_excel(file, header=0)
            matrix = df.values
            for i in range(len(matrix[:,0])):
                list_layers.append(matrix[i, 0])
                list_reg.append(matrix[i, 1])
                list_lambda.append(matrix[i, 2])
                list_nnode.append(matrix[i,3])
                list_epoch_of_min.append(matrix[i,4])
                list_tr_loss.append(matrix[i, 5])
                list_val_loss.append(matrix[i, 6])
    vec_layers = np.array(list_layers)
    vec_reg = np.array(list_reg)
    vec_lambda = np.array(list_lambda)
    vec_nnode = np.array(list_nnode)
    vec_nnode = np.array(list_nnode)
    vec_epoch_of_min = np.array(list_epoch_of_min)
    vec_tr_loss = np.array(list_tr_loss)
    vec_val_loss = np.array(list_val_loss)

    i = np.argmin(vec_val_loss)

    #Chosen hyperparameters
    chosen_lam = vec_lambda[i]
    chosen_nnode = vec_nnode[i]
    chosen_layers = vec_layers[i]
    chosen_reg = vec_reg[i]
    chosen_epoch = vec_epoch_of_min[i]

    return chosen_lam, chosen_nnode, chosen_layers, chosen_reg,chosen_epoch
