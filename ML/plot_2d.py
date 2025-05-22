import numpy as np
import matplotlib.pyplot as plt
import torch
from Hertz import *

def plot_2d_cpress_force_radii(model, subfolder_path, which_radius, Rxw, Ryw, Rxr, min_Rxw,max_Rxw,min_Ryw,max_Ryw, min_Rxr,max_Rxr, max_cpress, max_Fn, chosen_lam, chosen_nnode, chosen_layers, chosen_reg):
    
    '''
    Function to create 2d plot of the maximum contact pressure with regards to the force and a radius

    Parameters:

    model (nn.Neural): Neural network model created and trained for input data
    subfolder_path (string): Path to subfolder where figure is saved
    which_radius (string): Which radius that is plotted one of the axes
    Rxw (float or int): radius (in mm) of the wheel in x-z plane  
    Ryw (float or int): radius (in mm) of the wheel in y-z plane
    Rxr (float or int): radius (in mm) of the rail in x-z plane
    min_Rxw (float or int): The minimum value of radius (in mm) of the wheel in x-z plane. Same as the one used when creating data for model.
    max_Rxw (float or int): The maximum value of radius (in mm) of the wheel in x-z plane. Same as the one used when creating data for model.
    min_Ryw (float or int): The minimum value of radius (in mm) of the wheel in y-z plane. Same as the one used when creating data for model.
    max_Ryw (float or int): The maximum value of radius (in mm) of the wheel in y-z plane. Same as the one used when creating data for model.
    min_Rxr (float or int): The minimum value of radius (in mm) of the rail in x-z plane. Same as the one used when creating data for model.
    max_Rxr (float or int): The maximum value of radius (in mm) of the rail in x-z plane. Same as the one used when creating data for model.
    max_cpress (float or int): The maximum value of the output of the contact pressure in MPa, used for scaling
    max_Fn (float or int): The maximum value of the applied force. Same as the one used when creating data for model.
    chosen_lam (float or int): The chosen regularization hyperparameter for model
    chosen_nnode (int): The chosen number of nodes per hidden layer for model
    chosen_layers (int): The chosen number of hidden layers for model
    chosen_reg (string): The chosen regularization type ('Lasso' or 'Ridge')

    Returns:
    A 2d figure of maximum contact pressure versus force and a chosen radius for the model contra the data
    '''
    #Load training, validation, and test data that was used for model
    x_train = model.x_train
    y_train = model.y_train
    x_val = model.x_val
    y_val = model.y_val
    x_test = model.x_test
    y_test = model.y_test
    
    #Create a meshgrid of the force and radius
    if which_radius == 'Rxw':
        radius = np.linspace(min_Rxw*0.9, max_Rxw*1.1, 100)
    if which_radius == 'Ryw':
        radius = np.linspace(min_Ryw*0.9, max_Ryw*1.1, 100)
    if which_radius == 'Rxr':
        radius = np.linspace(min_Rxr*0.9, max_Rxr*1.1, 100)
    f = np.linspace(0, 1.25*max_Fn, 100)
    Force, Radius = np.meshgrid(f, radius)

    #Empty matrix for the output
    P = np.zeros(np.shape(Force))

    
    #Model prediction
    for i in range(len(f)):
        for j in range(len(radius)):
            if which_radius == 'Rxw':
                Rxw = Radius[i,j]
            if which_radius == 'Ryw':
                Ryw = Radius[i,j]
            if which_radius == 'Rxr':
                Rxr = Radius[i,j]
            Fn = Force[i,j]
            xin = np.array([[Rxw/max_Rxw, Ryw/max_Ryw, Rxr/max_Rxr, Fn/max_Fn]])
            out = model(torch.tensor(xin, dtype=torch.float32)).detach().numpy()
            P[i,j] = out[0,0]*max_cpress
    
    #Saving values of training data for 2d plot for the chosen geometry
    F_train = []
    P_train = []
    R_train = []
    for i in range(len(x_train[:,0])):
        if which_radius == 'Rxw':
            if x_train[i,1] == Ryw/max_Ryw:
                if x_train[i,2] == Rxr/max_Rxr:
                    F_train.append(x_train[i,3]*max_Fn)
                    P_train.append(y_train[i,0]*max_cpress)
                    R_train.append(x_train[i, 0]*max_Rxw)
        if which_radius == 'Ryw':
            if x_train[i,0] == Rxw/max_Rxw:
                if x_train[i,2] == Rxr/max_Rxr:
                    F_train.append(x_train[i,3]*max_Fn)
                    P_train.append(y_train[i,0]*max_cpress)
                    R_train.append(x_train[i, 1]*max_Ryw)
        if which_radius == 'Rxr':
            if x_train[i,0] == Rxw/max_Rxw:
                if x_train[i,1] == Ryw/max_Ryw:
                    F_train.append(x_train[i,3]*max_Fn)
                    P_train.append(y_train[i,0]*max_cpress)
                    R_train.append(x_train[i, 2]*max_Rxr)
    F_train = np.array(F_train)

    #Saving values of validation data for 2d plot for the chosen geometry
    F_val = []
    P_val = []
    R_val = []
    for i in range(len(x_val[:,0])):
        if which_radius == 'Rxw':
            if x_val[i,1] == Ryw/max_Ryw:
                if x_val[i,2] == Rxr/max_Rxr:
                    F_val.append(x_val[i,3]*max_Fn)
                    P_val.append(y_val[i,0]*max_cpress)
                    R_val.append(x_val[i, 0]*max_Rxw)
        if which_radius == 'Ryw':
            if x_val[i,0] == Rxw/max_Rxw:
                if x_val[i,2] == Rxr/max_Rxr:
                    F_val.append(x_val[i,3]*max_Fn)
                    P_val.append(y_val[i,0]*max_cpress)
                    R_val.append(x_val[i, 1]*max_Ryw)
        if which_radius == 'Rxr':
            if x_val[i,0] == Rxw/max_Rxw:
                if x_val[i,1] == Ryw/max_Ryw:
                    F_val.append(x_val[i,3]*max_Fn)
                    P_val.append(y_val[i,0]*max_cpress)
                    R_val.append(x_val[i, 2]*max_Rxr)
    F_val = np.array(F_val)

    #Saving values of test data for 2d plot for the chosen geometry
    F_test = []
    P_test = []
    R_test = []
    for i in range(len(x_test[:,0])):
        if which_radius == 'Rxw':
            if x_test[i,1] == Ryw/max_Ryw:
                if x_test[i,2] == Rxr/max_Rxr:
                    F_test.append(x_test[i,3]*max_Fn)
                    P_test.append(y_test[i,0]*max_cpress)
                    R_test.append(x_test[i, 0]*max_Rxw)
        if which_radius == 'Ryw':
            if x_test[i,0] == Rxw/max_Rxw:
                if x_test[i,2] == Rxr/max_Rxr:
                    F_test.append(x_test[i,3]*max_Fn)
                    P_test.append(y_test[i,0]*max_cpress)
                    R_test.append(x_test[i, 1]*max_Ryw)
        if which_radius == 'Rxr':
            if x_test[i,0] == Rxw/max_Rxw:
                if x_test[i,1] == Ryw/max_Ryw:
                    F_test.append(x_test[i,3]*max_Fn)
                    P_test.append(y_test[i,0]*max_cpress)
                    R_test.append(x_test[i, 2]*max_Rxr)
    F_test = np.array(F_test)

    # Plotting
    fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"}, label='Model')
    ax1.plot_surface(Force*1e-3, Radius, P, alpha=0.5)
    ax1.scatter(F_train*1e-3, R_train, P_train, color='red', label='Training data')
    ax1.scatter(F_val*1e-3, R_val, P_val, color='black', label='Validation data')
    ax1.scatter(F_test*1e-3, R_test, P_test, color='green', label='Test data')
    ax1.set_xlabel(r'$F_n$ [kN]', fontsize=15)
    ax1.set_zlabel(r'$p_{max}$ [MPa]', fontsize=15)
    if which_radius == 'Rxw':
        ax1.set_ylabel(r'$R_{xw}$ [mm]', fontsize=15)
    if which_radius == 'Ryw':
        ax1.set_ylabel(r'$R_{yw}$ [mm]', fontsize=15)
    if which_radius == 'Rxr':
        ax1.set_ylabel(r'$R_{xr}$ [mm]', fontsize=15)    
    ax1.legend(fontsize=12)
    fig1.tight_layout()
    if which_radius == 'Rxw':
        fig1.savefig(subfolder_path+'/Force_vs_Rxw'+f'_Rwy={Ryw}_Rrx={Rxr}_layers={chosen_layers}_nodes={chosen_nnode}_lam={chosen_lam}_reg='+chosen_reg+'.pdf')
    if which_radius == 'Ryw':
        fig1.savefig(subfolder_path+'/Force_vs_Ryw'+f'_Rwx={Rxw}_Rrx={Rxr}_layers={chosen_layers}_nodes={chosen_nnode}_lam={chosen_lam}_reg='+chosen_reg+'.pdf')
    if which_radius == 'Rxr':
        fig1.savefig(subfolder_path+'/Force_vs_Rxr'+f'_Rwx={Rxw}_Rwy={Ryw}_layers={chosen_layers}_nodes={chosen_nnode}_lam={chosen_lam}_reg='+chosen_reg+'.pdf')
    plt.close(fig1)
    return

def plot_2d_vonMises_force_radii(model, subfolder_path, which_radius, Rxw, Ryw, Rxr, min_Rxw,max_Rxw,min_Ryw,max_Ryw, min_Rxr,max_Rxr, max_vM, max_Fn, chosen_lam, chosen_nnode, chosen_layers, chosen_reg):
    '''
    Function to create 2d plot of the maximum contact pressure with regards to the force and a radius

    Parameters:

    model (nn.Neural): Neural network model created and trained for input data
    subfolder_path (string): Path to subfolder where figure is saved
    which_radius (string): Which radius that is plotted one of the axes
    Rxw (float or int): radius (in mm) of the wheel in x-z plane  
    Ryw (float or int): radius (in mm) of the wheel in y-z plane
    Rxr (float or int): radius (in mm) of the rail in x-z plane
    min_Rxw (float or int): The minimum value of radius (in mm) of the wheel in x-z plane. Same as the one used when creating data for model.
    max_Rxw (float or int): The maximum value of radius (in mm) of the wheel in x-z plane. Same as the one used when creating data for model.
    min_Ryw (float or int): The minimum value of radius (in mm) of the wheel in y-z plane. Same as the one used when creating data for model.
    max_Ryw (float or int): The maximum value of radius (in mm) of the wheel in y-z plane. Same as the one used when creating data for model.
    min_Rxr (float or int): The minimum value of radius (in mm) of the rail in x-z plane. Same as the one used when creating data for model.
    max_Rxr (float or int): The maximum value of radius (in mm) of the rail in x-z plane. Same as the one used when creating data for model.
    max_vM (float or int): The maximum value of the output of the von Mises stress in MPa, used for scaling
    max_Fn (float or int): The maximum value of the applied force. Same as the one used when creating data for model.
    chosen_lam (float or int): The chosen regularization hyperparameter for model
    chosen_nnode (int): The chosen number of nodes per hidden layer for model
    chosen_layers (int): The chosen number of hidden layers for model
    chosen_reg (string): The chosen regularization type ('Lasso' or 'Ridge')

    Returns:
    A 2d figure of maximum von Mises stress versus force and a chosen radius for the model contra the data
    '''
    #Load training, validation, and test data that was used for model
    x_train = model.x_train
    y_train = model.y_train
    x_val = model.x_val
    y_val = model.y_val
    x_test = model.x_test
    y_test = model.y_test
    
    #Create a meshgrid of the force and radius
    if which_radius == 'Rxw':
        radius = np.linspace(min_Rxw*0.9, max_Rxw*1.1, 100)
    if which_radius == 'Ryw':
        radius = np.linspace(min_Ryw*0.9, max_Ryw*1.1, 100)
    if which_radius == 'Rxr':
        radius = np.linspace(min_Rxr*0.9, max_Rxr*1.1, 100)
    f = np.linspace(0, 1.25*max_Fn, 100)
    Force, Radius = np.meshgrid(f, radius)

    #Empty matrix for the output
    vM = np.zeros(np.shape(Force))

    #Model prediction
    for i in range(len(f)):
        for j in range(len(radius)):
            if which_radius == 'Rxw':
                Rxw = Radius[i,j]
            if which_radius == 'Ryw':
                Ryw = Radius[i,j]
            if which_radius == 'Rxr':
                Rxr = Radius[i,j]
            Fn = Force[i,j]
            xin = np.array([[Rxw/max_Rxw, Ryw/max_Ryw, Rxr/max_Rxr, Fn/max_Fn]])
            out = model(torch.tensor(xin, dtype=torch.float32)).detach().numpy()
            vM[i,j] = out[0,2]*max_vM
    
    #Saving values of training data for 2d plot for the chosen geometry
    F_train = []
    vM_train = []
    R_train = []
    for i in range(len(x_train[:,0])):
        if which_radius == 'Rxw':
            if x_train[i,1] == Ryw/max_Ryw:
                if x_train[i,2] == Rxr/max_Rxr:
                    F_train.append(x_train[i,3]*max_Fn)
                    vM_train.append(y_train[i,2]*max_vM)
                    R_train.append(x_train[i, 0]*max_Rxw)
        if which_radius == 'Ryw':
            if x_train[i,0] == Rxw/max_Rxw:
                if x_train[i,2] == Rxr/max_Rxr:
                    F_train.append(x_train[i,3]*max_Fn)
                    vM_train.append(y_train[i,2]*max_vM)
                    R_train.append(x_train[i, 1]*max_Ryw)
        if which_radius == 'Rxr':
            if x_train[i,0] == Rxw/max_Rxw:
                if x_train[i,1] == Ryw/max_Ryw:
                    F_train.append(x_train[i,3]*max_Fn)
                    vM_train.append(y_train[i,2]*max_vM)
                    R_train.append(x_train[i, 2]*max_Rxr)
    F_train = np.array(F_train)
    
    #Saving values of validation data for 2d plot for the chosen geometry
    F_val = []
    vM_val = []
    R_val = []
    for i in range(len(x_val[:,0])):
        if which_radius == 'Rxw':
            if x_val[i,1] == Ryw/max_Ryw:
                if x_val[i,2] == Rxr/max_Rxr:
                    F_val.append(x_val[i,3]*max_Fn)
                    vM_val.append(y_val[i,2]*max_vM)
                    R_val.append(x_val[i, 0]*max_Rxw)
        if which_radius == 'Ryw':
            if x_val[i,0] == Rxw/max_Rxw:
                if x_val[i,2] == Rxr/max_Rxr:
                    F_val.append(x_val[i,3]*max_Fn)
                    vM_val.append(y_val[i,2]*max_vM)
                    R_val.append(x_val[i, 1]*max_Ryw)
        if which_radius == 'Rxr':
            if x_val[i,0] == Rxw/max_Rxw:
                if x_val[i,1] == Ryw/max_Ryw:
                    F_val.append(x_val[i,3]*max_Fn)
                    vM_val.append(y_val[i,2]*max_vM)
                    R_val.append(x_val[i, 2]*max_Rxr)
    F_val = np.array(F_val)
    
    #Saving values of test data for 2d plot for the chosen geometry
    F_test = []
    vM_test = []
    R_test = []
    for i in range(len(x_test[:,0])):
        if which_radius == 'Rxw':
            if x_test[i,1] == Ryw/max_Ryw:
                if x_test[i,2] == Rxr/max_Rxr:
                    F_test.append(x_test[i,3]*max_Fn)
                    vM_test.append(y_test[i,2]*max_vM)
                    R_test.append(x_test[i, 0]*max_Rxw)
        if which_radius == 'Ryw':
            if x_test[i,0] == Rxw/max_Rxw:
                if x_test[i,2] == Rxr/max_Rxr:
                    F_test.append(x_test[i,3]*max_Fn)
                    vM_test.append(y_test[i,2]*max_vM)
                    R_test.append(x_test[i, 1]*max_Ryw)
        if which_radius == 'Rxr':
            if x_test[i,0] == Rxw/max_Rxw:
                if x_test[i,1] == Ryw/max_Ryw:
                    F_test.append(x_test[i,3]*max_Fn)
                    vM_test.append(y_test[i,2]*max_vM)
                    R_test.append(x_test[i, 2]*max_Rxr)
    F_test = np.array(F_test)

    # Plotting
    fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
    ax1.plot_surface(Force*1e-3, Radius, vM, alpha=0.5, label='Model')
    ax1.scatter(F_train*1e-3, R_train, vM_train, color='red', label='Training data')
    ax1.scatter(F_val*1e-3, R_val, vM_val, color='black', label='Validation data')
    ax1.scatter(F_test*1e-3, R_test, vM_test, color='green', label='Test data')
    ax1.set_xlabel(r'$F_n$ [kN]', fontsize=15)
    ax1.set_zlabel(r'Maximum von Mises stress [MPa]', fontsize=15)
    if which_radius == 'Rxw':
        ax1.set_ylabel(r'$R_{xw}$ [mm]', fontsize=15)
    if which_radius == 'Ryw':
        ax1.set_ylabel(r'$R_{yw}$ [mm]', fontsize=15)
    if which_radius == 'Rxr':
        ax1.set_ylabel(r'$R_{xr}$ [mm]', fontsize=15)
    ax1.tick_params(axis='both', which='major', labelsize=12)    
    ax1.legend(fontsize=12)
    fig1.tight_layout()
    if which_radius == 'Rxw':
        fig1.savefig(subfolder_path+'/Force_vs_Rxw'+f'_Rwy={Ryw}_Rrx={Rxr}_layers={chosen_layers}_nodes={chosen_nnode}_lam={chosen_lam}_reg='+chosen_reg+'.pdf')
    if which_radius == 'Ryw':
        fig1.savefig(subfolder_path+'/Force_vs_Ryw'+f'_Rwx={Rxw}_Rrx={Rxr}_layers={chosen_layers}_nodes={chosen_nnode}_lam={chosen_lam}_reg='+chosen_reg+'.pdf')
    if which_radius == 'Rxr':
        fig1.savefig(subfolder_path+'/Force_vs_Rxr'+f'_Rwx={Rxw}_Rwy={Ryw}_layers={chosen_layers}_nodes={chosen_nnode}_lam={chosen_lam}_reg='+chosen_reg+'.pdf')
    plt.close(fig1)
    return


def plot_2d_disp_force_radii(model, subfolder_path, which_radius, Rxw, Ryw, Rxr, min_Rxw,max_Rxw,min_Ryw,max_Ryw, min_Rxr,max_Rxr, max_disp, max_Fn, chosen_lam, chosen_nnode, chosen_layers, chosen_reg):

    '''
    Function to create 2d plot of the maximum vertical displacement with regards to the force and a radius

    Parameters:

    model (nn.Neural): Neural network model created and trained for input data
    subfolder_path (string): Path to subfolder where figure is saved
    which_radius (string): Which radius that is plotted one of the axes
    Rxw (float or int): radius (in mm) of the wheel in x-z plane  
    Ryw (float or int): radius (in mm) of the wheel in y-z plane
    Rxr (float or int): radius (in mm) of the rail in x-z plane
    min_Rxw (float or int): The minimum value of radius (in mm) of the wheel in x-z plane. Same as the one used when creating data for model.
    max_Rxw (float or int): The maximum value of radius (in mm) of the wheel in x-z plane. Same as the one used when creating data for model.
    min_Ryw (float or int): The minimum value of radius (in mm) of the wheel in y-z plane. Same as the one used when creating data for model.
    max_Ryw (float or int): The maximum value of radius (in mm) of the wheel in y-z plane. Same as the one used when creating data for model.
    min_Rxr (float or int): The minimum value of radius (in mm) of the rail in x-z plane. Same as the one used when creating data for model.
    max_Rxr (float or int): The maximum value of radius (in mm) of the rail in x-z plane. Same as the one used when creating data for model.
    max_disp (float or int): The maximum value of the output of the displacement in mm, used for scaling
    max_Fn (float or int): The maximum value of the applied force. Same as the one used when creating data for model.
    chosen_lam (float or int): The chosen regularization hyperparameter for model
    chosen_nnode (int): The chosen number of nodes per hidden layer for model
    chosen_layers (int): The chosen number of hidden layers for model
    chosen_reg (string): The chosen regularization type ('Lasso' or 'Ridge')

    Returns:
    A 2d figure of maximum vertical displacement versus force and a chosen radius for the model contra the data
    '''

    #Load training, validation, and test data that was used for model
    x_train = model.x_train
    y_train = model.y_train
    x_val = model.x_val
    y_val = model.y_val
    x_test = model.x_test
    y_test = model.y_test
    
    #Create a meshgrid of the force and radius
    if which_radius == 'Rxw':
        radius = np.linspace(min_Rxw*0.9, max_Rxw*1.1, 100)
    if which_radius == 'Ryw':
        radius = np.linspace(min_Ryw*0.9, max_Ryw*1.1, 100)
    if which_radius == 'Rxr':
        radius = np.linspace(min_Rxr*0.9, max_Rxr*1.1, 100)
    f = np.linspace(0, 1.25*max_Fn, 100)
    Force, Radius = np.meshgrid(f, radius)

    #Empty matrix for the output
    Disp = np.zeros(np.shape(Force))

    #Model prediction
    for i in range(len(f)):
        for j in range(len(radius)):
            if which_radius == 'Rxw':
                Rxw = Radius[i,j]
            if which_radius == 'Ryw':
                Ryw = Radius[i,j]
            if which_radius == 'Rxr':
                Rxr = Radius[i,j]
            Fn = Force[i,j]
            xin = np.array([[Rxw/max_Rxw, Ryw/max_Ryw, Rxr/max_Rxr, Fn/max_Fn]])
            out = model(torch.tensor(xin, dtype=torch.float32)).detach().numpy()
            Disp[i,j] = out[0,1]*max_disp
    
    #Saving values of training data for 2d plot for the chosen geometry
    F_train = []
    Disp_train = []
    R_train = []
    for i in range(len(x_train[:,0])):
        if which_radius == 'Rxw':
            if x_train[i,1] == Ryw/max_Ryw:
                if x_train[i,2] == Rxr/max_Rxr:
                    F_train.append(x_train[i,3]*max_Fn)
                    Disp_train.append(y_train[i,1]*max_disp)
                    R_train.append(x_train[i, 0]*max_Rxw)
        if which_radius == 'Ryw':
            if x_train[i,0] == Rxw/max_Rxw:
                if x_train[i,2] == Rxr/max_Rxr:
                    F_train.append(x_train[i,3]*max_Fn)
                    Disp_train.append(y_train[i,1]*max_disp)
                    R_train.append(x_train[i, 1]*max_Ryw)
        if which_radius == 'Rxr':
            if x_train[i,0] == Rxw/max_Rxw:
                if x_train[i,1] == Ryw/max_Ryw:
                    F_train.append(x_train[i,3]*max_Fn)
                    Disp_train.append(y_train[i,1]*max_disp)
                    R_train.append(x_train[i, 2]*max_Rxr)
    F_train = np.array(F_train)

    #Saving values of validation data for 2d plot for the chosen geometry
    F_val = []
    Disp_val = []
    R_val = []
    for i in range(len(x_val[:,0])):
        if which_radius == 'Rxw':
            if x_val[i,1] == Ryw/max_Ryw:
                if x_val[i,2] == Rxr/max_Rxr:
                    F_val.append(x_val[i,3]*max_Fn)
                    Disp_val.append(y_val[i,1]*max_disp)
                    R_val.append(x_val[i, 0]*max_Rxw)
        if which_radius == 'Ryw':
            if x_val[i,0] == Rxw/max_Rxw:
                if x_val[i,2] == Rxr/max_Rxr:
                    F_val.append(x_val[i,3]*max_Fn)
                    Disp_val.append(y_val[i,1]*max_disp)
                    R_val.append(x_val[i, 1]*max_Ryw)
        if which_radius == 'Rxr':
            if x_val[i,0] == Rxw/max_Rxw:
                if x_val[i,1] == Ryw/max_Ryw:
                    F_val.append(x_val[i,3]*max_Fn)
                    Disp_val.append(y_val[i,1]*max_disp)
                    R_val.append(x_val[i, 2]*max_Rxr)
    F_val = np.array(F_val)

    #Saving values of test data for 2d plot for the chosen geometry
    F_test = []
    Disp_test = []
    R_test = []
    for i in range(len(x_test[:,0])):
        if which_radius == 'Rxw':
            if x_test[i,1] == Ryw/max_Ryw:
                if x_test[i,2] == Rxr/max_Rxr:
                    F_test.append(x_test[i,3]*max_Fn)
                    Disp_test.append(y_test[i,1]*max_disp)
                    R_test.append(x_test[i, 0]*max_Rxw)
        if which_radius == 'Ryw':
            if x_test[i,0] == Rxw/max_Rxw:
                if x_test[i,2] == Rxr/max_Rxr:
                    F_test.append(x_test[i,3]*max_Fn)
                    Disp_test.append(y_test[i,1]*max_disp)
                    R_test.append(x_test[i, 1]*max_Ryw)
        if which_radius == 'Rxr':
            if x_test[i,0] == Rxw/max_Rxw:
                if x_test[i,1] == Ryw/max_Ryw:
                    F_test.append(x_test[i,3]*max_Fn)
                    Disp_test.append(y_test[i,1]*max_disp)
                    R_test.append(x_test[i, 2]*max_Rxr)
    F_test = np.array(F_test)

    # Plotting
    fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
    ax1.plot_surface(Force*1e-3, Radius, Disp, alpha=0.5, label='Model')
    ax1.scatter(F_train*1e-3, R_train, Disp_train, color='red', label='Training data')
    ax1.scatter(F_val*1e-3, R_val, Disp_val, color='black', label='Validation data')
    ax1.scatter(F_test*1e-3, R_test, Disp_test, color='green', label='Test data')
    ax1.set_xlabel(r'$F_n$ [kN]', fontsize=15)
    ax1.set_zlabel(r'Max vertical disp. [MPa]', fontsize=15)
    if which_radius == 'Rxw':
        ax1.set_ylabel(r'$R_{xw}$ [mm]', fontsize=15)
    if which_radius == 'Ryw':
        ax1.set_ylabel(r'$R_{yw}$ [mm]', fontsize=15)
    if which_radius == 'Rxr':
        ax1.set_ylabel(r'$R_{xr}$ [mm]', fontsize=15)
    ax1.tick_params(axis='both', which='major', labelsize=12)    
    ax1.legend(fontsize=12)
    fig1.tight_layout()
    if which_radius == 'Rxw':
        fig1.savefig(subfolder_path+'/Force_vs_Rxw'+f'_Rwy={Ryw}_Rrx={Rxr}_layers={chosen_layers}_nodes={chosen_nnode}_lam={chosen_lam}_reg='+chosen_reg+'.pdf')
    if which_radius == 'Ryw':
        fig1.savefig(subfolder_path+'/Force_vs_Ryw'+f'_Rwx={Rxw}_Rrx={Rxr}_layers={chosen_layers}_nodes={chosen_nnode}_lam={chosen_lam}_reg='+chosen_reg+'.pdf')
    if which_radius == 'Rxr':
        fig1.savefig(subfolder_path+'/Force_vs_Rxr'+f'_Rwx={Rxw}_Rwy={Ryw}_layers={chosen_layers}_nodes={chosen_nnode}_lam={chosen_lam}_reg='+chosen_reg+'.pdf')
    plt.close(fig1)
    return

