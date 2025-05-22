import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# With 4 inputs and 3 outputs
def NN3(x, y, lam, layers, split, seednum, reg, num_of_nodes,max_no_epoch = 100000, forcemult = True, blind = True, overtrain_softcond = True, conv_crit_on = True, overtrain_crit_on = True):
    '''
    Function for creating and training a neural network for a certain set of hyperparameters

    Parameters:
    x (np.array): Matrix with input data, needs to have four columns, the force is the last column
    y (np.array): Matrix with output data, needs to have three columns
    lam (float or int): Regularization hyperparameter
    layers (int): Number of hidden layers
    split (float): The fraction of the data that is training data, between 0 and 1
    seednum (int): The seednumber for the random choice of the division of data
    reg (float): Regularization type, 'Lasso' or 'Ridge'
    num_of_nodes (int): Number of nodes per hidden layer
    max_no_epoch (int): The maximum number of epochs for which the NN trains
    forcemult (logical): If True the output is multiplied by the input force
    blind (logical): If True the loss for the test data is not printed
    overtrain_softcond (logical): If True the model requires the validation loss to be 25% greater than the training loss for it to judge it as overtrained
    conv_crit_on (logical): If True it stops the training when the training loss goes below a threshold value
    overtrain_crit_on (logical): If True the training stops when the model has overtrained

    Returns:
    model (nn.Neural): The trained neural network model 
    train_losses[ind_min]: The training loss at the epoch of the minimum of the validation loss
    val_losses[ind_min]: The minimum validation loss
    epoch_of_min: The epoch of the minimum of the validation loss

    '''

    #Only one type of regularization
    lambda_l1 = (reg == 'Lasso')*lam
    lambda_l2 = (reg == 'Ridge')*lam
    
    np.random.seed(seednum)   # for getting out the same random variables every time (reproducability)

    ndata = len(x[:,0])# number of data points
    indices = np.random.permutation(ndata)
    split_index = int(ndata * split)
    x_train, x_nottrain = x[indices[:split_index],:], x[indices[split_index:],:]
    y_train, y_nottrain = y[indices[:split_index],:], y[indices[split_index:],:]

    # Splitting the data thats not training in half into validation and test data
    split_half = int(len(x_nottrain[:,0])*0.5)
    np.random.seed(seednum)
    indices = np.random.permutation(len(x_nottrain[:,0]))
    x_val, x_test = x_nottrain[indices[:split_half],:], x_nottrain[indices[split_half:],:]
    y_val, y_test = y_nottrain[indices[:split_half],:], y_nottrain[indices[split_half:],:]
    
    # Converting to Pytorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    
    torch.manual_seed(0) #Ensuring that initial model parameters are the same for each training
    #class for neural network
    class Neural(nn.Module):
        def __init__(self):
            super(Neural, self).__init__()
            # Four inputs and three outputs
            #Creates weights and biases between each layer
            self.f = {}
            self.f[f'fc1'] = nn.Linear(4, num_of_nodes)  # Fully connected layers
            for i in range(2, layers+1):
                self.f[f'fc{i}'] = nn.Linear(num_of_nodes,num_of_nodes)
            self.fclast = nn.Linear(num_of_nodes, 3)
            self.sigmoid = nn.Sigmoid()  # Sigmoid activation function
            #Saving what was training, validation and test data for the model
            self.x_train = x_train
            self.y_train = y_train
            self.x_val = x_val
            self.y_val = y_val
            self.x_test = x_test
            self.y_test = y_test

        def forward(self, x):
            forcein = x[:,3] #the applied force
            #Each layer are connected linearly and all hidden layers have sigmoid activation functions
            f1 = self.f.get(f'fc1')
            x = self.sigmoid(f1(x))
            for i in range(2, layers+1):
                fi = self.f.get(f'fc{i}')
                x = self.sigmoid(fi(x))
            x = self.fclast(x)
            if forcemult == True: #Multiplying final output by input force
                x[:,0] = torch.mul(forcein, x[:,0])
                x[:,1] = torch.mul(forcein, x[:,1])
                x[:,2] = torch.mul(forcein, x[:,2])
            return x
    # Creating an instance of the model
    model = Neural()

    #Model parameters
    d = torch.cat([param.view(-1) for param in model.parameters()])
    prev_d = d.clone().detach()

    torch.manual_seed(0)

    # Defining the loss function
    criterion = nn.MSELoss()

    def closure():
        optimizer.zero_grad()  # Zero out gradients
        outputs = model(x_train_tensor)  # Forward pass through the model
        loss= criterion(outputs, y_train_tensor)
        if reg == 'Lasso': 
            # Calculate the L1 regularization term
            l1_regularization = torch.tensor(0.)
            for param in model.parameters():
                l1_regularization += torch.linalg.vector_norm(param, ord=1)  # L1 norm of model parameters

            # Add the L1 regularization term to the loss
            loss += lambda_l1 * l1_regularization  # Add L1 regularization to the loss
        
        if reg == 'Ridge':
            # Calculate the L2  regularization term
            l2_regularization = torch.tensor(0.)
            for param in model.parameters():
                l2_regularization += torch.linalg.vector_norm(param, ord=2)**2  # L2 norm of model parameters

            # Add the L1 regularization term to the loss
            loss += lambda_l2 * l2_regularization  # Add L1 regularization to the loss
        
        loss.backward()  # Compute gradients
        return loss
    
    prev_loss = float('inf')  # Initializing with a large value
    tolerance_error = 1.e-9 #Threshold value for the loss change for convergence crit
    tolerance_param = 1.e-12 #Threshold value for when training stops due to little change of model parameters

    # Setting the optimizer 
    # lr is the initial learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    train_losses = []  # Store training losses
    val_losses = []   # Store validation losses
    test_losses = [] #Store test losses
    for epoch in range(max_no_epoch):
        def closure_wrapper():
            loss = closure() #return loss
            return loss
        
        loss = optimizer.step(closure_wrapper)  # Optimize model parameters

        # Evaluate the losses/MSE errors of validation, training and test
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            train_outputs = model(x_train_tensor)
            train_loss = criterion(train_outputs, y_train_tensor)
            test_outputs = model(x_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor)

        # Store training, validation, and test losses
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        test_losses.append(test_loss.item())
        
        # Print the loss every 1000th epoch
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{max_no_epoch}], Training Loss percent: {train_loss.item()*100:.15f}, Val Loss percent: {val_loss.item()*100:.15f}")

        #Computing change of model parameters
        d = torch.cat([param.view(-1) for param in model.parameters()])
        d_d=d.detach().numpy() - prev_d.detach().numpy()

        if np.sqrt( np.sum(d_d*d_d) ) < tolerance_param : 
            print('Change of optimization variables d_d is smaller than threshold')
            print(f"Epoch [{epoch + 1}/{max_no_epoch}], Training Loss percent: {train_loss.item()*100:.15f}, Val Loss percent: {val_loss.item()*100:.15f}")
            if blind == False:
                print(f"Test Loss percent: {test_loss.item()*100:.15f}")
            break

        #Computing loss change
        loss_change = abs(prev_loss - loss.item())
        prev_d = d.clone().detach()
        prev_loss = loss.item()

        # Check if the optimizer has converged (you can modify this condition)
        if conv_crit_on == True:
            if train_loss.item()<1.e-4:
                if loss_change < tolerance_error:
                    print(f"Optimizer has converged. Stopping training.")
                    print(f"Epoch [{epoch + 1}/{max_no_epoch}], Training Loss percent: {train_loss.item()*100:.15f}, Val Loss percent: {val_loss.item()*100:.15f}")
                    if blind == False:
                        print(f"Test Loss percent: {test_loss.item()*100:.15f}")
                    break

        #Check if model is overtraining
        if overtrain_crit_on == True:
            if overtrain_softcond == False:
                if epoch>100000: 
                    if val_loss.item()>train_loss.item():#Validation error greater than training
                        diff_testloss = np.diff(val_losses[-10000:])
                        sumdiff = np.sum(diff_testloss)
                        if sumdiff > 0: #Check if the validation loss has increased over past 10 thousand epochs
                            print(f"Increase in validation error. Stopping training.")
                            print(f"Epoch [{epoch + 1}/{max_no_epoch}], Training Loss percent: {train_loss.item()*100:.12f}, Val Loss percent: {val_loss.item()*100:.12f}")
                            if blind == False:
                                print(f"Test Loss percent: {test_loss.item()*100:.12f}")
                            break
            if overtrain_softcond == True:
                if epoch>100000:
                    # note the difference, higher val loss required to trigger
                    if val_loss.item()>1.25*train_loss.item(): #Validation error greater than training by 25%
                        diff_testloss = np.diff(val_losses[-10000:])
                        sumdiff = np.sum(diff_testloss)
                        if sumdiff > 0 :#Check if the validation loss has increased over past 10 thousand epochs
                            print(f"Increase in validation error. Stopping training.")
                            print(f"Epoch [{epoch + 1}/{max_no_epoch}], Training Loss percent: {train_loss.item()*100:.12f}, Val Loss percent: {val_loss.item()*100:.12f}")
                            if blind == False:
                                print(f"Test Loss percent: {test_loss.item()*100:.12f}")
                            break

    #Calculating the losses at the minimum of the validation loss
    ind_min = np.argmin(val_losses)
    epoch_of_min = ind_min+1
    print(f"The epoch of the minimum is [{epoch_of_min}/{max_no_epoch}], Training Loss percent: {train_losses[ind_min]*100:.12f}, Val Loss percent: {val_losses[ind_min]*100:.12f}")
    if blind == False:
        print(f"Test Loss percent: {test_losses[ind_min]*100:.12f}")

    #Plotting validation and training losses
    fig1, ax1 = plt.subplots()
    ax1.plot(np.arange(len(train_losses[1000:])), train_losses[1000:], 'b', label='Train loss')
    ax1.plot(np.arange(len(val_losses[1000:])), val_losses[1000:], 'r', label='With-hold loss')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('losses')
    ax1.set_yscale('log')
    ax1.legend()
    fig1.savefig('losses'+f'_layers={layers}_nodes={num_of_nodes}_lam={lam:.9f}_maxepoch={max_no_epoch}_reg='+reg+'.pdf')
    plt.close(fig1)



    return model, train_losses[ind_min], val_losses[ind_min], epoch_of_min