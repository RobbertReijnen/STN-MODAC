import os
import json
import multiprocessing

import numpy as np
from pathlib import Path

from run_GA import initialize_run
from config import BASE_PATH

folder_names = ['train/j5_m5', 'test/j5_m5']

def run(folder_name, pool):
    base_path = Path(__file__).parent.absolute()
    folder_path = base_path.joinpath(folder_name)
    lenght_path = len(folder_name.strip('/').split('/'))
    instances = ['/' + '/'.join(folder_path.parts[-lenght_path:]) + '/' + f for f in os.listdir(folder_path) if
                 os.path.isfile(os.path.join(folder_path, f))]

    new_reference_points = {}

    for instance in instances:
        print(instance)
        params = {"population_size": 1000, "problem_instance": instance, 'nr_of_objectives': 7}
        initial_population, toolbox, stats, hof, jobShopEnv = initialize_run(pool, **params)
        fitnesses = [ind.fitness.values for ind in initial_population]
        max_bounds = list(np.max(fitnesses, axis=0))
        new_reference_points[instance] = max_bounds

    pool.close()
    # Load existing reference points from the JSON file (if it exists)
    existing_reference_points = {}
    if os.path.isfile(BASE_PATH + '/datasets/reference_points_test.json'):
        with open(BASE_PATH + '/datasets/reference_points_test.json', 'r') as file:
            existing_reference_points = json.load(file)

    # Update the existing reference points with new ones, only adding new keys
    for key, value in new_reference_points.items():
        if key not in existing_reference_points:
            existing_reference_points[key] = value

    # Write the updated reference points back to the JSON file
    with open(BASE_PATH + '/datasets/reference_points_test.json', 'w') as file:
        json.dump(existing_reference_points, file)


if __name__ == '__main__':
    for folder_name in folder_names:
        print(folder_name)
        pool = multiprocessing.Pool()
        run(folder_name, pool)
