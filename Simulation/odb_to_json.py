from odbAccess import openOdb
import json

# Fetch and open the ODB and the step
generated_path = "MainJob.odb"
odb = openOdb(path = generated_path)

step_names = list(odb.steps.keys())
frame = odb.steps[step_names[-1]].frames[-1]

# CPRESS field
cpress_field = frame.fieldOutputs['CPRESS']
cpress_list = [value.data for value in cpress_field.values]
max_cpress = max(cpress_list)

# Stress field
stress = frame.fieldOutputs['S']
stress_list = [value.mises for value in stress.values]
max_vonmises = max(stress_list)

# Displacement field
u_field = frame.fieldOutputs['U']
u_list = [value.data[1] for value in u_field.values]    
min_u = min(u_list)

# Dump into json file
params = {
    "Max CPRESS":  str(max_cpress),
    "Max von Mises": str(max_vonmises),
    "Min displacement": str(min_u) 
}

with open(r"output_data.json", "w") as f:
    json.dump(params, f)