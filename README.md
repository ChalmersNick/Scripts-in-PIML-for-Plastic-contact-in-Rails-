# Welcome to this repository!
It contains scripts and datafiles connected to a project in the course TME131 Project in Applied Mechanics in the MSc in Applied Mechanics program at Chalmers University of Technology. The project regarded simulating training data with Abaqus to train an ML model that can predict different quantities (max von Mises stress, max contact pressure, etc.) regarding contact between a train wheel and a rail. The purpose was to make a faster calculation with the model instead of relying on heavy simulations for accurate results.

## Info about the code
The simulations were automated with Abaqus Python interpreter and the code for the simulation and reading of the data can be found here, along with the scripts that trained the ML model.

Note that the running the main script with all the combinations of input data could take a long time. A solution to this was to make copies of the "Simulation" folder and run the simulation with several subsets of the input data at the same time. There is no built-in functionality for that in the code but it is recommended since the training data generation would take very long otherwise. The reason for the need to make copies of the folder is that Abaqus creates lock files that makes sure that there aren't several simulations in the same directory. A consequence of running in different folders is the creation of several data files which will have to be considered when training the model later.

## Authors
The authors of the code are
- Melvin Glans
- William Gustafsson
- Nikhil Katti
- Axel Lindmark
- Marcus Petersson