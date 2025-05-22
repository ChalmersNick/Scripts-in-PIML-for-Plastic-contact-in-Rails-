import numpy as np
import matplotlib.pyplot as plt
import torch
from Hertz import * #imports Hertzian contact theory class

def plot_1d_cpress(model, subfolder_path, Rxw, Ryw, Rxr, max_Rxw, max_Ryw, max_Rxr, max_cpress, max_Fn, chosen_lam, chosen_nnode, chosen_layers, chosen_reg):
    
    '''
    Function to create 1d plot of the maximum contact pressure with regards to the force for a certain input geometry

    Parameters:

    model (nn.Neural): Neural network model created and trained for input data
    subfolder_path (string): Path to subfolder where figure is saved
    Rxw (float or int): radius (in mm) of the wheel in x-z plane  
    Ryw (float or int): radius (in mm) of the wheel in y-z plane
    Rxr (float or int): radius (in mm) of the rail in x-z plane
    max_Rxw (float or int): The maximum value of radius (in mm) of the wheel in x-z plane. Same as the one used when creating data for model.
    max_Ryw (float or int): The maximum value of radius (in mm) of the wheel in y-z plane. Same as the one used when creating data for model.
    max_Rxr (float or int): The maximum value of radius (in mm) of the rail in x-z plane. Same as the one used when creating data for model.
    max_cpress (float or int): The maximum value of the output of the contact pressure in MPa, used for scaling
    max_Fn (float or int): The maximum value of the applied force. Same as the one used when creating data for model.
    chosen_lam (float or int): The chosen regularization hyperparameter for model
    chosen_nnode (int): The chosen number of nodes per hidden layer for model
    chosen_layers (int): The chosen number of hidden layers for model
    chosen_reg (string): The chosen regularization type ('Lasso' or 'Ridge')

    Returns:
    A 1d figure of maximum contact pressure versus force for the model contra the data
    '''

    #Load training, validation, and test data that was used for model
    x_train = model.x_train
    y_train = model.y_train
    x_val = model.x_val
    y_val = model.y_val
    x_test = model.x_test
    y_test = model.y_test

    #Saving values of training data for 1d plot for the chosen geometry
    plot_p_train = []
    plot_F_train = []
    for i in range(len(x_train[:,0])):
        if x_train[i,0] == Rxw/max_Rxw:
            if x_train[i,1] == Ryw/max_Ryw:
                if x_train[i,2] == Rxr/max_Rxr:
                    plot_F_train.append(x_train[i,3])
                    plot_p_train.append(y_train[i,0])
    plot_p_train = np.array(plot_p_train)
    plot_F_train = np.array(plot_F_train)

    #Saving values of validation data for 1d plot for the chosen geometry
    plot_p_val = []
    plot_F_val = []
    for i in range(len(x_val[:,0])):
        if x_val[i,0] == Rxw/max_Rxw:
            if x_val[i,1] == Ryw/max_Ryw:
                if x_val[i,2] == Rxr/max_Rxr:
                    plot_F_val.append(x_val[i,3])
                    plot_p_val.append(y_val[i,0])
    plot_p_val = np.array(plot_p_val)
    plot_F_val = np.array(plot_F_val)

    #Saving values of test data for 1d plot for the chosen geometry
    plot_p_test = []
    plot_F_test = []
    for i in range(len(x_test[:,0])):
        if x_test[i,0] == Rxw/max_Rxw:
            if x_test[i,1] == Ryw/max_Ryw:
                if x_test[i,2] == Rxr/max_Rxr:
                    plot_F_test.append(x_test[i,3])
                    plot_p_test.append(y_test[i,0])
    plot_p_test = np.array(plot_p_test)
    plot_F_test = np.array(plot_F_test)

    #Saving values of the models prediction within a span
    plot_F_model = np.linspace(1e-9,1.25) #unscaled force span
    plot_p_model = []
    plot_Hertz = []
    for i in range(len(plot_F_model)):
        xin = np.array([[Rxw/max_Rxr, Ryw/max_Ryw, Rxr/max_Rxr, plot_F_model[i]]])
        out = model(torch.tensor(xin, dtype=torch.float32)).detach().numpy()
        plot_p_model.append(out[0,0])
        #Prediction of maximum contact pressure per Hertzian contact theory
        prediction = np.array(HertzClass([Rxw*1e-3, Ryw*1e-3, Rxr*1e-3, max_Fn*plot_F_model[i]*4], [200e9, 0.28, 200e9, 0.28]).predic())
        plot_Hertz.append(prediction*1e-6) 
    
    plot_p_model = np.array(plot_p_model)
    plot_Hertz = np.array(plot_Hertz)

    #Plotting
    fig1, ax1 = plt.subplots()
    ax1.plot(plot_F_model*max_Fn*1e-3, plot_p_model*max_cpress, label='Model')
    ax1.plot(plot_F_model*max_Fn*1e-3, plot_Hertz, label='Hertzian theory')
    ax1.scatter(plot_F_train*max_Fn*1e-3, plot_p_train*max_cpress, color='red', label='Training data')
    ax1.scatter(plot_F_val*max_Fn*1e-3, plot_p_val*max_cpress, color='black', label='Validation data')
    ax1.scatter(plot_F_test*max_Fn*1e-3, plot_p_test*max_cpress, color='green', label='Test data')
    ax1.set_ylabel(r'$p_{max}$ [MPa]', fontsize=18)
    ax1.set_xlabel(r'$F_n$ [kN]', fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax1.grid(True)
    ax1.legend(fontsize=15)
    fig1.tight_layout()
    fig1.savefig(subfolder_path+'/Force_vs_pmax'+f'_Rwx={Rxw}_Rwy={Ryw}_Rrx={Rxr}_layers={chosen_layers}_nodes={chosen_nnode}_lam={chosen_lam}_reg='+chosen_reg+'.pdf')
    plt.close(fig1)
    return

def plot_1d_vonMises(model, subfolder_path, Rxw, Ryw, Rxr, max_Rxw, max_Ryw, max_Rxr, max_vM, max_Fn, chosen_lam, chosen_nnode, chosen_layers, chosen_reg):

    '''
    Function to create 1d plot of the maximum von Mises stress with regards to the force for a certain input geometry

    Parameters:

    model (nn.Neural): Neural network model created and trained for input data
    subfolder_path (string): Path to subfolder where figure is saved
    Rxw (float or int): radius (in mm) of the wheel in x-z plane  
    Ryw (float or int): radius (in mm) of the wheel in y-z plane
    Rxr (float or int): radius (in mm) of the rail in x-z plane
    max_Rxw (float or int): The maximum value of radius (in mm) of the wheel in x-z plane. Same as the one used when creating data for model.
    max_Ryw (float or int): The maximum value of radius (in mm) of the wheel in y-z plane. Same as the one used when creating data for model.
    max_Rxr (float or int): The maximum value of radius (in mm) of the rail in x-z plane. Same as the one used when creating data for model.
    max_vM (float or int): The maximum value of the output of the von Mises stress in MPa, used for scaling
    max_Fn (float or int): The maximum value of the applied force. Same as the one used when creating data for model.
    chosen_lam (float or int): The chosen regularization hyperparameter for model
    chosen_nnode (int): The chosen number of nodes per hidden layer for model
    chosen_layers (int): The chosen number of hidden layers for model
    chosen_reg (string): The chosen regularization type ('Lasso' or 'Ridge')

    Returns:
    A 1d figure of maximum von Mises stress versus force for the model contra the data
    '''

    #Load training, validation, and test data that was used for model
    x_train = model.x_train
    y_train = model.y_train
    x_val = model.x_val
    y_val = model.y_val
    x_test = model.x_test
    y_test = model.y_test

    #Saving values of training data for 1d plot for the chosen geometry
    plot_vM_train = []
    plot_F_train = []
    for i in range(len(x_train[:,0])):
        if x_train[i,0] == Rxw/max_Rxw:
            if x_train[i,1] == Ryw/max_Ryw:
                if x_train[i,2] == Rxr/max_Rxr:
                    plot_F_train.append(x_train[i,3])
                    plot_vM_train.append(y_train[i,2])
    plot_vM_train = np.array(plot_vM_train)
    plot_F_train = np.array(plot_F_train)

    #Saving values of validation data for 1d plot for the chosen geometry
    plot_vM_val = []
    plot_F_val = []
    for i in range(len(x_val[:,0])):
        if x_val[i,0] == Rxw/max_Rxw:
            if x_val[i,1] == Ryw/max_Ryw:
                if x_val[i,2] == Rxr/max_Rxr:
                    plot_F_val.append(x_val[i,3])
                    plot_vM_val.append(y_val[i,2])
    plot_vM_val = np.array(plot_vM_val)
    plot_F_val = np.array(plot_F_val)

    #Saving values of test data for 1d plot for the chosen geometry
    plot_vM_test = []
    plot_F_test = []
    for i in range(len(x_test[:,0])):
        if x_test[i,0] == Rxw/max_Rxw:
            if x_test[i,1] == Ryw/max_Ryw:
                if x_test[i,2] == Rxr/max_Rxr:
                    plot_F_test.append(x_test[i,3])
                    plot_vM_test.append(y_test[i,2])
    plot_vM_test = np.array(plot_vM_test)
    plot_F_test = np.array(plot_F_test)

    #Saving values of the models prediction within a span
    plot_F_model = np.linspace(0,1.25)
    plot_vM_model = []
    for i in range(len(plot_F_model)):
        xin = np.array([[Rxw/max_Rxr, Ryw/max_Ryw, Rxr/max_Rxr, plot_F_model[i]]])
        out = model(torch.tensor(xin, dtype=torch.float32)).detach().numpy()
        plot_vM_model.append(out[0,2])
    
    plot_vM_model = np.array(plot_vM_model)

    #Plotting
    fig1, ax1 = plt.subplots()
    ax1.plot(plot_F_model*max_Fn*1e-3, plot_vM_model*max_vM, label='Model')
    ax1.scatter(plot_F_train*max_Fn*1e-3, plot_vM_train*max_vM, color='red', label='Training data')
    ax1.scatter(plot_F_val*max_Fn*1e-3, plot_vM_val*max_vM, color='black', label='Validation data')
    ax1.scatter(plot_F_test*max_Fn*1e-3, plot_vM_test*max_vM, color='green', label='Test data')
    ax1.set_ylabel(r'Maximum von Mises stress [MPa]', fontsize=18)
    ax1.set_xlabel(r'$F_n$ [kN]', fontsize=18)
    ax1.grid(True)
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax1.legend(fontsize=15)
    fig1.tight_layout()
    fig1.savefig(subfolder_path+'/Force_vs_vM'+f'_Rwx={Rxw}_Rwy={Ryw}_Rrx={Rxr}_layers={chosen_layers}_nodes={chosen_nnode}_lam={chosen_lam}_reg='+chosen_reg+'.pdf')
    plt.close(fig1)
    return

def plot_1d_disp(model, subfolder_path, Rxw, Ryw, Rxr, max_Rxw, max_Ryw, max_Rxr, max_disp, max_Fn, chosen_lam, chosen_nnode, chosen_layers, chosen_reg):

    '''
    Function to create 1d plot of the maximum contact pressure with regards to the force for a certain input geometry

    Parameters:

    model (nn.Neural): Neural network model created and trained for input data
    subfolder_path (string): Path to subfolder where figure is saved
    Rxw (float or int): radius (in mm) of the wheel in x-z plane  
    Ryw (float or int): radius (in mm) of the wheel in y-z plane
    Rxr (float or int): radius (in mm) of the rail in x-z plane
    max_Rxw (float or int): The maximum value of radius (in mm) of the wheel in x-z plane. Same as the one used when creating data for model.
    max_Ryw (float or int): The maximum value of radius (in mm) of the wheel in y-z plane. Same as the one used when creating data for model.
    max_Rxr (float or int): The maximum value of radius (in mm) of the rail in x-z plane. Same as the one used when creating data for model.
    max_disp (float or int): The maximum value of the output of the displacement in mm, used for scaling
    max_Fn (float or int): The maximum value of the applied force. Same as the one used when creating data for model.
    chosen_lam (float or int): The chosen regularization hyperparameter for model
    chosen_nnode (int): The chosen number of nodes per hidden layer for model
    chosen_layers (int): The chosen number of hidden layers for model
    chosen_reg (string): The chosen regularization type ('Lasso' or 'Ridge')

    Returns:
    A 1d figure of maximum vertical displacement versus force for the model contra the data
    '''

    #Load training, validation, and test data that was used for model
    x_train = model.x_train
    y_train = model.y_train
    x_val = model.x_val
    y_val = model.y_val
    x_test = model.x_test
    y_test = model.y_test

    #Saving values of training data for 1d plot for the chosen geometry
    plot_disp_train = []
    plot_F_train = []
    for i in range(len(x_train[:,0])):
        if x_train[i,0] == Rxw/max_Rxw:
            if x_train[i,1] == Ryw/max_Ryw:
                if x_train[i,2] == Rxr/max_Rxr:
                    plot_F_train.append(x_train[i,3])
                    plot_disp_train.append(y_train[i,1])
    plot_disp_train = np.array(plot_disp_train)
    plot_F_train = np.array(plot_F_train)

    #Saving values of validation data for 1d plot for the chosen geometry
    plot_disp_val = []
    plot_F_val = []
    for i in range(len(x_val[:,0])):
        if x_val[i,0] == Rxw/max_Rxw:
            if x_val[i,1] == Ryw/max_Ryw:
                if x_val[i,2] == Rxr/max_Rxr:
                    plot_F_val.append(x_val[i,3])
                    plot_disp_val.append(y_val[i,1])
    plot_disp_val = np.array(plot_disp_val)
    plot_F_val = np.array(plot_F_val)

    #Saving values of displacement data for 1d plot for the chosen geometry
    plot_disp_test = []
    plot_F_test = []
    for i in range(len(x_test[:,0])):
        if x_test[i,0] == Rxw/max_Rxw:
            if x_test[i,1] == Ryw/max_Ryw:
                if x_test[i,2] == Rxr/max_Rxr:
                    plot_F_test.append(x_test[i,3])
                    plot_disp_test.append(y_test[i,1])
    plot_disp_test = np.array(plot_disp_test)
    plot_F_test = np.array(plot_F_test)

    #Saving values of the models prediction within a span
    plot_F_model = np.linspace(0,1.25)
    plot_disp_model = []
    for i in range(len(plot_F_model)):
        xin = np.array([[Rxw/max_Rxr, Ryw/max_Ryw, Rxr/max_Rxr, plot_F_model[i]]])
        out = model(torch.tensor(xin, dtype=torch.float32)).detach().numpy()
        plot_disp_model.append(out[0,1])
    plot_disp_model = np.array(plot_disp_model)

    #Plotting
    fig1, ax1 = plt.subplots()
    ax1.plot(plot_F_model*max_Fn*1e-3, plot_disp_model*max_disp, label='Model')
    ax1.scatter(plot_F_train*max_Fn*1e-3, plot_disp_train*max_disp, color='red', label='Training data')
    ax1.scatter(plot_F_val*max_Fn*1e-3, plot_disp_val*max_disp, color='black', label='Validation data')
    ax1.scatter(plot_F_test*max_Fn*1e-3, plot_disp_test*max_disp, color='green', label='Test data')
    ax1.set_ylabel(r'Max vertical disp. [mm]', fontsize=18)
    ax1.set_xlabel(r'$F_n$ [kN]', fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax1.grid(True)
    ax1.legend(fontsize=15)
    fig1.tight_layout()
    fig1.savefig(subfolder_path+'/Force_vs_disp'+f'_Rwx={Rxw}_Rwy={Ryw}_Rrx={Rxr}_layers={chosen_layers}_nodes={chosen_nnode}_lam={chosen_lam}_reg='+chosen_reg+'.pdf')
    plt.close(fig1)
    return