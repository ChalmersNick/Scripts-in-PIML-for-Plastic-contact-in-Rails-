import numpy as np
import os
import h5py

from neural_network import NN3 #imports the neural network function
from read_excelsheet import read #import function to read the files of different hyperparameters
from plot_1d import * #imports functions for 1d plotting of neural network model
from plot_2d import * #imports 2d plotting functions
from Hertz import * #imports Hertzian contact theory class


#Creates figures subfolders if one does not exist 
# Get the current working directory
current_directory = os.getcwd()

# Define the name of the subfolders for figures
subfolder_name_cpress_1d = "Figures/cpress/1d"
subfolder_name_disp_1d = "Figures/disp/1d"
subfolder_name_vonMises_1d = "Figures/vonMises/1d"
subfolder_name_cpress_2d = "Figures/cpress/2d"
subfolder_name_disp_2d = "Figures/disp/2d"
subfolder_name_vonMises_2d = "Figures/vonMises/2d"

# Create the full path for the subfolders
subfolder_path_cpress_1d = os.path.join(current_directory, subfolder_name_cpress_1d)
subfolder_path_disp_1d = os.path.join(current_directory, subfolder_name_disp_1d)
subfolder_path_vonMises_1d = os.path.join(current_directory, subfolder_name_vonMises_1d)
subfolder_path_cpress_2d = os.path.join(current_directory, subfolder_name_cpress_2d)
subfolder_path_disp_2d = os.path.join(current_directory, subfolder_name_disp_2d)
subfolder_path_vonMises_2d = os.path.join(current_directory, subfolder_name_vonMises_2d)

# Create the subfolders
os.makedirs(subfolder_path_cpress_1d, exist_ok=True)
os.makedirs(subfolder_path_disp_1d, exist_ok=True)
os.makedirs(subfolder_path_vonMises_1d, exist_ok=True)
os.makedirs(subfolder_path_cpress_2d, exist_ok=True)
os.makedirs(subfolder_path_disp_2d, exist_ok=True)
os.makedirs(subfolder_path_vonMises_2d, exist_ok=True)


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

ndata = len(wheelradiusX)

#Maximum values for scaling
max_Rxw = np.max(wheelradiusX)
max_Ryw = np.max(wheelradiusY)
max_Rxr = np.max(railradius)
max_Fn = np.max(axleload)

max_cpress = np.max(cpress)
max_disp = np.max(disp)
max_vM = np.max(vonMises)

#Minimum values for plotting
min_Rxw = np.min(wheelradiusX)
min_Ryw = np.min(wheelradiusY)
min_Rxr = np.min(railradius)

#Assembling input data and output data into scaled matrixes
x = np.array([wheelradiusX/max_Rxw, wheelradiusY/max_Ryw, railradius/max_Rxr, axleload/max_Fn]).T
y= np.array([cpress/max_cpress, disp/max_disp, vonMises/max_vM]).T

chosen_lam, chosen_nnode, chosen_layers, chosen_reg,chosen_epoch = read() #From the grid searches read the best hyperparameters
print(chosen_lam, chosen_nnode, chosen_layers, chosen_reg,chosen_epoch)

#Run the training with the ''best'' hyperparameters to the epoch of the minimum validation loss
model, trLoss, valLoss, epoch_of_min =NN3(x,y,chosen_lam, chosen_layers, 0.8, 0, chosen_reg, chosen_nnode,max_no_epoch=chosen_epoch,forcemult=True, blind = False,overtrain_softcond=False,conv_crit_on=False,overtrain_crit_on=False)

#Plotting the relationship between the force and the outputs for 10 different geometries, for the final model
for i in np.arange(0,len(wheelradiusX), 400):
    Rxw = wheelradiusX[i]
    Ryw = wheelradiusY[i]
    Rxr = railradius[i]
    plot_1d_cpress(model, subfolder_path_cpress_1d, Rxw, Ryw, Rxr, max_Rxw, max_Ryw, max_Rxr, max_cpress, max_Fn, chosen_lam, chosen_nnode, chosen_layers, chosen_reg)
    plot_1d_disp(model, subfolder_path_disp_1d, Rxw, Ryw, Rxr, max_Rxw, max_Ryw, max_Rxr, max_disp, max_Fn, chosen_lam, chosen_nnode, chosen_layers, chosen_reg)
    plot_1d_vonMises(model, subfolder_path_vonMises_1d, Rxw, Ryw, Rxr, max_Rxw, max_Ryw, max_Rxr, max_vM, max_Fn, chosen_lam, chosen_nnode, chosen_layers, chosen_reg)

#Creates one 2d plot for each of the outputs with the force and one of the radii
Rxw = wheelradiusX[2000]
Ryw = wheelradiusY[2000]
Rxr = railradius[2000]
plot_2d_cpress_force_radii(model, subfolder_path_cpress_2d, 'Rxw', Rxw, Ryw, Rxr, min_Rxw,max_Rxw,min_Ryw,max_Ryw, min_Rxr,max_Rxr, max_cpress, max_Fn, chosen_lam, chosen_nnode, chosen_layers, chosen_reg)
plot_2d_cpress_force_radii(model, subfolder_path_cpress_2d, 'Ryw', Rxw, Ryw, Rxr, min_Rxw,max_Rxw,min_Ryw,max_Ryw, min_Rxr,max_Rxr, max_cpress, max_Fn, chosen_lam, chosen_nnode, chosen_layers, chosen_reg)
plot_2d_cpress_force_radii(model, subfolder_path_cpress_2d, 'Rxr', Rxw, Ryw, Rxr, min_Rxw,max_Rxw,min_Ryw,max_Ryw, min_Rxr,max_Rxr, max_cpress, max_Fn, chosen_lam, chosen_nnode, chosen_layers, chosen_reg)
plot_2d_vonMises_force_radii(model, subfolder_path_vonMises_2d, 'Rxw', Rxw, Ryw, Rxr, min_Rxw,max_Rxw,min_Ryw,max_Ryw, min_Rxr,max_Rxr, max_vM, max_Fn, chosen_lam, chosen_nnode, chosen_layers, chosen_reg)
plot_2d_vonMises_force_radii(model, subfolder_path_vonMises_2d, 'Ryw', Rxw, Ryw, Rxr, min_Rxw,max_Rxw,min_Ryw,max_Ryw, min_Rxr,max_Rxr, max_vM, max_Fn, chosen_lam, chosen_nnode, chosen_layers, chosen_reg)
plot_2d_vonMises_force_radii(model, subfolder_path_vonMises_2d, 'Rxr', Rxw, Ryw, Rxr, min_Rxw,max_Rxw,min_Ryw,max_Ryw, min_Rxr,max_Rxr, max_vM, max_Fn, chosen_lam, chosen_nnode, chosen_layers, chosen_reg)
plot_2d_disp_force_radii(model, subfolder_path_disp_2d, 'Rxw', Rxw, Ryw, Rxr, min_Rxw,max_Rxw,min_Ryw,max_Ryw, min_Rxr,max_Rxr, max_disp, max_Fn, chosen_lam, chosen_nnode, chosen_layers, chosen_reg)
plot_2d_disp_force_radii(model, subfolder_path_disp_2d, 'Ryw', Rxw, Ryw, Rxr, min_Rxw,max_Rxw,min_Ryw,max_Ryw, min_Rxr,max_Rxr, max_disp, max_Fn, chosen_lam, chosen_nnode, chosen_layers, chosen_reg)
plot_2d_disp_force_radii(model, subfolder_path_disp_2d, 'Rxr', Rxw, Ryw, Rxr, min_Rxw,max_Rxw,min_Ryw,max_Ryw, min_Rxr,max_Rxr, max_disp, max_Fn, chosen_lam, chosen_nnode, chosen_layers, chosen_reg)