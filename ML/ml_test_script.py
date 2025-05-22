import numpy as np
import os
import pandas as pd
import h5py

from neural_network import NN3 #imports the neural network function


#This is a script for grid searching of neural network

#Load data from h5 files
axleload_list = []
railradius_list = []
wheelradiusX_list = []
wheelradiusY_list = []
cpress_list = []
disp_list = []
vonMises_list = []
for file in os.listdir(os.curdir):
    if file.endswith(".h5"):
        with h5py.File(file,'r') as f:
            group_keys = list(f.keys())
            n = 1
            for jobs in group_keys:
                number = f'0{n}'*(n < 10) + f'{n}'*(n >= 10)
                if n>10:
                    number = jobs[4:]
                axleload_list.append(f[str(jobs)+f'/input data {number}/AxleLoad{number}'][0])
                railradius_list.append(f[str(jobs)+f'/input data {number}/RailRadius{number}'][0])
                wheelradiusX_list.append(f[str(jobs)+f'/input data {number}/WheelRadiusX{number}'][0])
                wheelradiusY_list.append(f[str(jobs)+f'/input data {number}/WheelRadiusY{number}'][0])
                cpress_list.append(f[str(jobs)+f'/output data {number}/Max CPRESS{number}'][0])
                disp_list.append(f[str(jobs)+f'/output data {number}/Min displacement{number}'][0])
                vonMises_list.append(f[str(jobs)+f'/output data {number}/Max von Mises{number}'][0])
                n += 1
axleload = np.array(axleload_list)
railradius = np.array(railradius_list)
wheelradiusX = np.array(wheelradiusX_list)
wheelradiusY = np.array(wheelradiusY_list)
cpress = np.array(cpress_list)
disp = np.array(disp_list)
vonMises = np.array(vonMises_list)

#Assembling input data and output data into scaled matrixes
x = np.array([wheelradiusX/np.max(wheelradiusX), wheelradiusY/np.max(wheelradiusY), railradius/np.max(railradius), axleload/np.max(axleload)]).T
y= np.array([cpress/np.max(cpress), disp/np.max(disp), vonMises/np.max(vonMises)]).T

#Reading hyperparameters from txt file
#Can be changed to 'hyperparametersLasso.txt' or 'hyperparametersRidge.txt'
inpdata = pd.read_csv('hyperparameters_noreg.txt', sep = ',')

#File path to output
file_path ='hyperparamtest.xlsx'

#Initializing lists for storage of relevant data 
list_trLoss = []
list_valLoss = []
list_epoch_of_min = []
global_list_lam = []
global_list_node_per_layer = []
global_list_layers = []
global_list_reg = []

#Index to start and end reading through the list
low_index = int(input("Enter lower index: "))
high_index = int(input("Enter higher index: "))

################## Loop that runs the script starts here ##################
for i in inpdata.index[low_index:high_index]: #change index
    layers,reg,lam,nnode = inpdata.iloc[i, :]
    model, trLoss, valLoss, epoch_of_min =NN3(x,y,lam, layers, 0.8, 0, reg, nnode,max_no_epoch=int(1e6),forcemult=True, blind = True,overtrain_softcond=True,conv_crit_on=False,overtrain_crit_on=True)

    #Saving to lists
    list_trLoss.append(trLoss)
    list_valLoss.append(valLoss)
    list_epoch_of_min.append(epoch_of_min)
    global_list_lam.append(lam)
    global_list_node_per_layer.append(nnode)
    global_list_layers.append(layers)
    global_list_reg.append(reg)

    # New row to append
    new_row = {'Layers': layers,
                'Regularization type': reg, 
                'lambda': lam,
                '#Nodes': nnode,
                'Epoch of min': epoch_of_min,  
                'Training loss': trLoss, 
                'Validation loss': valLoss}
    
    if os.path.exists(file_path): #File already exists
        # Read existing data
        df_existing = pd.read_excel(file_path)

        # Append new data
        df_combined = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save to Excel
        df_combined.to_excel(file_path, index=False)
    else:
        df = pd.DataFrame([new_row])
        # Save to Excel
        df.to_excel(file_path, index=False)