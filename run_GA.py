import os
import argparse
import logging
import json
import multiprocessing
import numpy as np
import pickle
from deap import base, creator, tools
from multiprocessing.pool import Pool

from helper_functions_main import compute_hypervolume
from scheduling.plotting.drawer import draw_precedence_relations, draw_gantt_chart
from scheduling.helper_functions import record_stats, load_parameters, load_job_shop_env, save_results

from scheduling.genetic_algorithm.operators import (init_population, init_individual, evaluate_population,
                                                    evaluate_individual, mutate_shortest_proc_time,
                                                    mutate_sequence_exchange, variation,
                                                    pox_crossover, repair_precedence_constraints)


logging.basicConfig(level=logging.INFO)

from config import BASE_PATH

PARAM_FILE = BASE_PATH + "/configs/GA.json"
DEFAULT_RESULTS_ROOT = "./results/single_runs"
REFERENCE_POINTS_FILE = BASE_PATH + "/datasets/reference_points.json"


def compute_objectives(hof, kwargs):
    # Load existing reference point and compute hypervolume
    if os.path.isfile(REFERENCE_POINTS_FILE):
        with open(REFERENCE_POINTS_FILE, 'r') as file:
            reference_points = json.load(file)
            if kwargs['problem_instance'] in reference_points:
                reference_point = reference_points[kwargs['problem_instance']][0:kwargs['nr_of_objectives']]
                hypervolume = compute_hypervolume(hof, kwargs['nr_of_objectives'], list(reference_point))
                kwargs['hypervolume'] = hypervolume
            else:
                #print('NO REFERENCE POINT KNOWN, using [0,0,...,0] as reference point.')
                reference_point = [0 for i in range(kwargs['nr_of_objectives'])]
                hypervolume = compute_hypervolume(hof, kwargs['nr_of_objectives'], list(reference_point))
                kwargs['hypervolume'] = hypervolume
    else:
        #print('NO REFERENCE POINT KNOWN, using [0,0,...,0] as reference point.')
        reference_point = [0 for i in range(kwargs['nr_of_objectives'])]
        hypervolume = compute_hypervolume(hof, kwargs['nr_of_objectives'], list(reference_point))
        kwargs['hypervolume'] = hypervolume

    kwargs['hof'] = hof
    return kwargs


def initialize_run(pool: Pool, **kwargs):
    """Initializes the run by setting up the environment, toolbox, statistics, hall of fame, and initial population.

    Args:
        pool: Multiprocessing pool.
        kwargs: Additional keyword arguments.

    Returns:
        A tuple containing the initial population, toolbox, statistics, hall of fame, and environment.
    """
    if 'FJSP_SRMN' in kwargs['problem_instance']:
        try:
            with open(BASE_PATH + 'datasets' + kwargs["problem_instance"], 'rb') as f:
                jobShopEnv = pickle.load(f)
        except FileNotFoundError:
            logging.error(f"Problem instance {kwargs['problem_instance']} not found.")
            return
    else:
        try:
            jobShopEnv = load_job_shop_env(kwargs['problem_instance'])
        except FileNotFoundError:
            logging.error(f"Problem instance {kwargs['problem_instance']} not found.")
            return

    toolbox = base.Toolbox()
    if pool != None:
        toolbox.register("map", pool.map)

    creator.create("Fitness", base.Fitness, weights=tuple([-1.0 for i in range(kwargs['nr_of_objectives'])]))
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox.register("init_individual", init_individual, creator.Individual, jobShopEnv=jobShopEnv)
    toolbox.register("mate_TwoPoint", tools.cxTwoPoint)
    toolbox.register("mate_Uniform", tools.cxUniform, indpb=0.5)
    toolbox.register("mate_POX", pox_crossover, nr_preserving_jobs=1)

    toolbox.register("mutate_machine_selection", mutate_shortest_proc_time, jobShopEnv=jobShopEnv)
    toolbox.register("mutate_operation_sequence", mutate_sequence_exchange)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate_individual", evaluate_individual, jobShopEnv=jobShopEnv, objectives=kwargs['nr_of_objectives'])

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    hof = tools.ParetoFront()

    initial_population = init_population(toolbox, kwargs['population_size'], )
    try:
        fitnesses = evaluate_population(toolbox, initial_population, kwargs['nr_of_objectives'], logging)
    except Exception as e:
        logging.error(f"An error occurred during initial population evaluation: {e}")
        return

    for ind, fit in zip(initial_population, fitnesses):
        ind.fitness.values = fit

    return initial_population, toolbox, stats, hof, jobShopEnv


def run_algo(jobShopEnv, population, toolbox, folder, exp_name, stats=None, hof=None, **kwargs):
    """Executes the genetic algorithm and returns the best individual.

    Args:
        jobShopEnv: The problem environment.
        population: The initial population.
        toolbox: DEAP toolbox.
        folder: The folder to save results in.
        exp_name: The experiment name.
        stats: DEAP statistics (optional).
        hof: Hall of Fame (optional).
        kwargs: Additional keyword arguments.

    Returns:
        The best individual found by the genetic algorithm.
    """

    if kwargs['plotting']:
        if 'FJSP_SRMN' not in kwargs['problem_instance']:
            draw_precedence_relations(jobShopEnv)

    hof.update(population)

    gen = 0
    df_list = []
    logbook = tools.Logbook()
    logbook.header = ["gen"] + (stats.fields if stats else [])

    # Update the statistics with the new population
    record_stats(gen, population, logbook, stats, kwargs['logbook'], df_list, logging)

    if kwargs['logbook']:
        logging.info(logbook.stream)

    for gen in range(1, kwargs['ngen'] + 1):
        # Vary the population
        offspring = variation(population, toolbox, kwargs['population_size'], kwargs['cr'], kwargs['indpb'])

        # Ensure that precedence constraints between jobs are satisfied (only for assembly scheduling (fajsp))
        if '/dafjs/' in kwargs['problem_instance'] or '/yfjs/' in kwargs['problem_instance']:
            offspring = repair_precedence_constraints(jobShopEnv, offspring)

        # Evaluate the population
        fitnesses = evaluate_population(toolbox, offspring, kwargs['nr_of_objectives'], logging)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        hof.update(offspring)

        # Select next generation population
        population[:] = toolbox.select(population + offspring, kwargs['population_size'])
        # Update the statistics with the new population
        record_stats(gen, population, logbook, stats, kwargs['logbook'], df_list, logging)

    kwargs = compute_objectives(hof, kwargs)

    if kwargs['plotting']:
        objectives, jobShopEnv = evaluate_individual(hof[0], jobShopEnv, kwargs['nr_of_objectives'], alt_objectives=kwargs['alternative_objectives'], reset=False)
        draw_gantt_chart(jobShopEnv)

    if folder != None:
        save_results(hof, logbook, folder, exp_name, kwargs)

    return kwargs['hypervolume']


def main(param_file=PARAM_FILE):
    try:
        parameters = load_parameters(param_file)
    except FileNotFoundError:
        logging.error(f"Parameter file {param_file} not found.")
        return
    
    pool = multiprocessing.Pool()
    folder = (
            DEFAULT_RESULTS_ROOT
            + "/"
            + str(parameters['problem_instance'])
            + "/ngen"
            + str(parameters["ngen"])
            + "_pop"
            + str(parameters['population_size'])
            + "_cr"
            + str(parameters["cr"])
            + "_indpb"
            + str(parameters["indpb"])
    )

    exp_name = ("rseed" + str(parameters["rseed"]))
    population, toolbox, stats, hof, jobShopEnv = initialize_run(pool, **parameters)
    run_algo(jobShopEnv, population, toolbox, folder, exp_name, stats, hof, **parameters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GA")
    parser.add_argument(
        "config_file",
        metavar='-f',
        type=str,
        nargs="?",
        default=PARAM_FILE,
        help="path to config file",
    )

    args = parser.parse_args()
    main(param_file=args.config_file)
