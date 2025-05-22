import os
import json
import h5py
import numpy as np
import time
import pandas as pd
import subprocess

results_dir = 'MainOutput' # Path for results folder (ODB)
os.makedirs(results_dir, exist_ok=True) # Creates results folder
sim_script = 'FixedModel.py' # Name of script to run
params_file = "input_params.json"

mesh_value = 1.5

overwrite = input('Type "1" to overwrite inputdata.txt: ') == '1'
if not os.path.exists('inputdata.txt') or overwrite:
    subprocess.run('python create_txt.py', shell = True)
inpdata = pd.read_csv('inputdata.txt', sep = ',')

n = 1 # Counter for #jobs
ind1 = int(input('Start index: '))
ind2 = int(input('End index: '))

if not os.path.exists('done_ind.txt'):
    with open('done_ind.txt', 'w') as f:
        f.write('Index 1-Index 2\n' + '0-0\n')
done = pd.read_csv('done_ind.txt', sep = '-')
assert(ind1 == max(float(done['Index 2'])) + 1)
assert(ind2 <= inpdata.index[-1])

with open('done_ind.txt', 'a') as f:
    f.write(f'{ind1}-{ind2 - 1}\n')

for i in inpdata.index[ind1:ind2]:
    r_rail, r_wheelx, r_wheely, load = inpdata.iloc[i, :]
    start = time.time()
    params = { # Write parameter file
        "JobName": f"MainJob",
        'MeshVal': mesh_value,
        "RailRadius": r_rail,
        "WheelRadiusX": r_wheelx,
        "WheelRadiusY": r_wheely,
        "AxleLoad": load
    }

    with open(params_file, "w") as f: # Write the params to a .json file for import to script
        json.dump(params, f)

    subprocess.run('abaqus cae noGUI=' + sim_script, shell = True)
    print(f'-----------------------------------------------------------------> Abaqus Job {n} finished!')

    subprocess.run('abaqus python ReadODBtoJson.py', shell = True)

    with open('output_data.json', 'r') as f:
        outdata = json.load(f)
    
    if not os.path.exists('data.h5') or n == 1:
        act = 'w'
    else:
        act = 'a'

    with h5py.File('data.h5', act) as f:
        m = f'0{n}'*(n < 10) + f'{n}'*(n >= 10)
        grp = f.create_group(f'job {m}')
        inputsub = grp.create_group(f'input data {m}')
        for i in range(2, len(params)):
            key = list(params.keys())[i] + str(m)
            val = abs(list(params.values())[i])
            inputsub.create_dataset(key, data = np.array([val]))
        outputsub = grp.create_group(f'output data {m}')
        for i in range(len(outdata)):
            key = list(outdata.keys())[i] + str(m)
            val = abs(float(list(outdata.values())[i]))
            outputsub.create_dataset(key, data = np.array([val]))
    n += 1
    print('-----------------------------------------------------------------> Loop time = ', time.time() - start)

subprocess.run(f'move {params['JobName']}* {results_dir}', shell = True)
subprocess.run(f'move abaqus* {results_dir}', shell = True)

print('-----------------------------------------------------------------> Script finished!')