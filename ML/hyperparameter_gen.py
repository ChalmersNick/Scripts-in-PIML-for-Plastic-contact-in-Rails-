#Create a txt file for the hyperparameters with no regularization/lambda = 0
list_layers = [1, 2,3, 4, 5]
list_lam = [0]
list_reg = ['Lasso']
list_node_per_layer = [5, 10, 30]

lines = ['#Layers, Reg, Lambda, #Nodes/layer\n']
for reg in list_reg:
    for lam in list_lam:
        for layer in list_layers:
            for nodes in list_node_per_layer:
                lines.append(f'{layer},{reg},{lam},{nodes}\n')

with open('hyperparameters_noreg.txt', 'w') as f:
    f.writelines(lines)


#Creates a txt files of the hyperparameters for Lasso regularization
list_layers = [1, 2,3, 4, 5,]
list_lam = [1e-9, 1e-6, 1e-3]
list_reg = ['Lasso']
list_node_per_layer = [5, 10, 30]
lines = ['#Layers, Reg, Lambda, #Nodes/layer\n']
for reg in list_reg:
    for lam in list_lam:
        for layer in list_layers:
            for nodes in list_node_per_layer:
                lines.append(f'{layer},{reg},{lam},{nodes}\n')

with open('hyperparametersLasso.txt', 'w') as f:
    f.writelines(lines)
    

#Creates a txt files of the hyperparameters for Lasso regularization
list_layers = [1, 2, 3, 4, 5]
list_lam = [1e-9, 1e-6, 1e-3]
list_reg = ['Ridge']
list_node_per_layer = [5, 10, 30]
lines = ['#Layers, Reg, Lambda, #Nodes/layer\n']
for reg in list_reg:
    for lam in list_lam:
        for layer in list_layers:
            for nodes in list_node_per_layer:
                lines.append(f'{layer},{reg},{lam},{nodes}\n')

with open('hyperparametersRidge.txt', 'w') as f:
    f.writelines(lines)

