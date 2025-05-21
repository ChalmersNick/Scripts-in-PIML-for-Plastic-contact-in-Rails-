from odbAccess import openOdb
import json
from pathlib import Path


# Fetch and open the ODB and the step


generated_path = "MainJob.odb"
odb = openOdb(path=generated_path)

#odb = openOdb(path=r'MainOutput\\MainJob.odb')
step_names = list(odb.steps.keys())
frame = odb.steps[step_names[-1]].frames[-1]

# CPRESS field
cpress_field = frame.fieldOutputs['CPRESS']

cpress_list = []
for value in cpress_field.values:
    cpress_list.append(value.data)

max_cpress = max(cpress_list)

# Stress field
stress = frame.fieldOutputs['S']

stress_list = []
for value in stress.values:
    vm = value.mises  # This is the von Mises stress
    stress_list.append(vm)

max_vonmises = max(stress_list)

# Displacement field
u_field = frame.fieldOutputs['U']

u_list = []
for value in u_field.values:
    u2 = value.data[1]  # U2 is the second component (index 1)
    u_list.append(u2)
    
    #print(f"Node: {value.nodeLabel}, U2: {u2}") kan få ut vilken nod oxå om man vill

min_u = min(u_list)

# Dump into json file
params = {
    "Max CPRESS":  str(max_cpress),
    "Max von Mises": str(max_vonmises),
    "Min displacement": str(min_u) 
}

with open(r"output_data.json", "w") as f:
    json.dump(params, f)