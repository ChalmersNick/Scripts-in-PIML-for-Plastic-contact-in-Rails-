import numpy as np

# Defining ranges for training data
rrlo, rrhi = 250, 350 # rail radius
rwxlo, rwxhi = 700, 1000 # wheel profile radius
rwylo, rwyhi = 980/2, 1300/2 # wheel size/main radius
loadlo, loadhi = 44*9.82e3/8/4, 21/8*9.82e3 # load

n = 8

# sampling radii wrt curvature instead of radius
rhors = np.linspace(1/rrhi, 1/rrlo, n)
rhowxs = np.linspace(1/rwxhi, 1/rwxlo, n)
rhowys = np.linspace(1/rwyhi, 1/rwylo, n)
loads = -np.linspace(loadhi, loadlo, n)

# converting back to get relevant input data for ML model
rrs = 1/rhors
rwxs = 1/rhowxs
rwys = 1/rhowys

# Starting with header
lines = ['Rail radius,Wheel radius x,Wheel radius y,Load\n']

# Looping through combinations of the input parameters to be added to txt file
for rr in rrs:
    for rwx in rwxs:
        for rwy in rwys:
            for load in loads:
                lines.append(f'{rr:.8f},{rwx:.8f},{rwy:.8f},{load:.6f}\n')

# Writing lines to txt file
with open('inputdata.txt', 'w') as f:
    f.writelines(lines)