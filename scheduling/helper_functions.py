import os
import json

import pandas as pd
import pathlib
from scheduling.scheduling_environment.jobShop import JobShop
from scheduling.data_parsers import parser_fjsp, parser_fajsp_sdsts, parser_fajsp


def load_parameters(config_json):
    """Load parameters from a json file"""
    with open(config_json, "rb") as f:
        config_params = json.load(f)
    return config_params


def load_job_shop_env(problem_instance: str, from_absolute_path=False) -> JobShop:

    jobShopEnv = JobShop()
    if '/fajsp_sdsts/' in problem_instance:
        jobShopEnv = parser_fajsp_sdsts.parse(jobShopEnv, problem_instance, from_absolute_path)
    elif '/FAJSP/' in problem_instance:
        jobShopEnv = parser_fajsp.parse(jobShopEnv, problem_instance, from_absolute_path)
    elif 'FJSP' in problem_instance or 'train' in problem_instance:
        jobShopEnv = parser_fjsp.parse(jobShopEnv, problem_instance, from_absolute_path)
    else:
        raise NotImplementedError(
            f"""Problem instance {
            problem_instance
            } not implemented"""
        )
    jobShopEnv._name = problem_instance
    return jobShopEnv


def create_stats_list(population, gen):
    stats_list = []
    for ind in population:
        tmp_dict = {}
        tmp_dict.update(
            {
                "Generation": gen,
                "obj1": ind.fitness.values[0]
            })
        if hasattr(ind, "objectives"):
            tmp_dict.update(
                {
                    "obj1": ind.objectives[0],
                }
            )
        tmp_dict = {**tmp_dict}
        stats_list.append(tmp_dict)
    return stats_list


def record_stats(gen, population, logbook, stats, verbose, df_list, logging):
    stats_list = create_stats_list(population, gen)
    df_list.append(pd.DataFrame(stats_list))
    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=gen, **record)
    if verbose:
        logging.info(logbook.stream)


def update_operations_available_for_scheduling(env):
    scheduled_operations = set(env.scheduled_operations)
    precedence_relations = env.precedence_relations_operations
    operations_available = [
        operation
        for operation in env.operations
        if operation not in scheduled_operations and all(
            prec_operation in scheduled_operations
            for prec_operation in precedence_relations[operation.operation_id]
        )
    ]
    env.set_operations_available_for_scheduling(operations_available)


def dict_to_excel(dictionary, folder, filename):
    """Save outputs in files"""

    # Check if the folder exists, if not, create it
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Check if the file exists, if so, give a warning
    full_path = os.path.join(folder, filename)
    if os.path.exists(full_path):
        print(f"Warning: {full_path} already exists. Overwriting file.")

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame([dictionary])

    # Save the DataFrame to Excel
    df.to_excel(full_path, index=False, engine='openpyxl')


def save_results(hof, logbook, folder, exp_name, kwargs):
    output_dir = folder
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Ensure that exp_name includes a slash (/) if needed
    exp_name = exp_name.strip("/")
    # Specify the full path for the logbook CSV file
    logbook_csv_path = os.path.join(output_dir, f'{exp_name}_logbook.csv')
    logbook_df = pd.DataFrame(logbook)
    logbook_df.to_csv(logbook_csv_path, index=False)

    # Create a DataFrame for the hall of fame data
    hof_data = []
    for ind in hof:
        hof_data.append(ind.fitness.values)

    hof_df = pd.DataFrame(hof_data, columns=[f'Objective_{i + 1}' for i in range(len(hof_data[0]))])
    hof_csv_path = os.path.join(output_dir, f'{exp_name}_hof.csv')
    hof_df.to_csv(hof_csv_path, index=False)

    # add best solution objectives to the parameters
    for i in range(len(hof[0].fitness.values)):
        kwargs['min_obj_{}'.format(i)] = min([ind.fitness.values[i] for ind in hof])
        kwargs['max_obj_{}'.format(i)] = max([ind.fitness.values[i] for ind in hof])

    # kwargs['hof'] = [i.fitness.values for i in hof]
    #
    # if len(hof[0].fitness.values) == 2:
    #     with open('/hpc/za64617/projects/GNN4APC_dev/code/src/scheduling/data/approximated_fronts_2_obj.json', 'r') as file:
    #         approximated_fronts = json.load(file)
    #
    # elif len(hof[0].fitness.values) == 3:
    #     with open('/hpc/za64617/projects/GNN4APC_dev/code/src/scheduling/data/approximated_fronts_3_obj.json', 'r') as file:
    #         approximated_fronts = json.load(file)
    #
    # elif len(hof[0].fitness.values) == 5:
    #     with open('/hpc/za64617/projects/GNN4APC_dev/code/src/scheduling/data/approximated_fronts_5_obj.json', 'r') as file:
    #         approximated_fronts = json.load(file)
    #
    # approximated_front = approximated_fronts[kwargs['problem_instance']]
    # #
    # print('compute alternative objectives')
    #
    # igd = IGD(np.array(approximated_front))
    # kwargs['igd'] = igd.do(np.array([list(item.fitness.values) for item in hof]))
    # print('igd: ', kwargs['igd'])
    # igd_plus = IGDPlus(np.array(approximated_front))
    # kwargs['igd_plus'] = igd_plus.do(np.array([list(item.fitness.values) for item in hof]))
    # kwargs['nr_solutions_hof'] = len(hof)

    results_csv_path = os.path.join(output_dir, f'{exp_name}_results.csv')
    df = pd.DataFrame.from_dict(kwargs, orient='index').T
    pd.DataFrame(df).to_csv(results_csv_path,index=False)

def select_first_operation_for_scheduling(operation):
    return operation.operation_options[0]

