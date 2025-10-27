import os
import csv
import json
import pandas as pd

directory_path = "/hpc/st-ds/projects/autoDAC_EA/za64617/ideal_points_alternative_problems/run_GA/"
objectives = ['min_obj_0', 'min_obj_1', 'min_obj_2', 'min_obj_3', 'min_obj_4', 'min_obj_5', 'min_obj_6', 'hypervolume']


list_of_files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f)) and 'results.csv' in f]

# append dfs to create one df with all results (results_df)
appended_dfs = []
for filename in list_of_files:
    df = pd.read_csv(directory_path + filename, index_col=None, header=0)
    appended_dfs.append(df)

results_df = pd.concat(appended_dfs, axis=0, ignore_index=True)
instances = results_df['problem_instance'].unique()
print(instances)
new_ideal_points = {}
for instance in sorted(instances):
    results = results_df[(results_df['problem_instance'] == instance)]
    new_ideal_points[instance] = [results['min_obj_0'].min(), results['min_obj_1'].min(), results['min_obj_2'].min(), results['min_obj_3'].min(), results['min_obj_4'].min()] #, results['min_obj_5'].min(), results['min_obj_6'].min()]


# Load existing reference points from the JSON file (if it exists)
existing_ideal_points = {}
if os.path.isfile('ideal_points_5_obj.json'):
    with open('ideal_points_5_obj.json', 'r') as file:
        existing_ideal_points = json.load(file)

# Update the existing reference points with new ones, only adding new keys
for key, value in new_ideal_points.items():
    if key not in existing_ideal_points:
        existing_ideal_points[key] = value

# Write the updated reference points back to the JSON file
with open('ideal_points_5_obj.json', 'w') as file:
    json.dump(existing_ideal_points, file)
