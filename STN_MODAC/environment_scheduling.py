import torch
import json
import pickle
import toml
import os

import pathlib
import matplotlib.pyplot as plt
plt.rcParams.update({'xtick.labelsize': 24, 'ytick.labelsize': 24})
import networkx as nx
import gymnasium as gym
import pandas as pd

from gymnasium import spaces
from deap import base, creator, tools

from helper_functions_main import compute_hypervolume

from scheduling.genetic_algorithm.operators import *
from scheduling.helper_functions import load_job_shop_env
from scheduling.genetic_algorithm.operators import (evaluate_individual, variation,
                                                    init_individual, init_population, mutate_shortest_proc_time,
                                                    mutate_sequence_exchange, pox_crossover)

from STN_MODAC.shared_memory import create_tensor_file, read_from_tensor_file, remove_tensor_file
from STN_MODAC.helper_functions import (
    get_front,
    normalize_graph_nodes,
    get_fitness_bounds,
    update_bounds
)

from config import BASE_PATH

REFERENCE_POINTS_FILE = BASE_PATH + "/datasets/reference_points.json"


def get_training_instances(problem_size, instance_range):
    base_path = f"/FJSP/train/{problem_size}/train_{problem_size}_"
    return [f"{base_path}{i}.txt" for i in range(instance_range[0], instance_range[1])]


class schedulingEnv(gym.Env):
    def __init__(self, parameters): #, embeddig_tracker):
        self.population_size = parameters['environment']['population_size']
        self.nr_objectives = parameters['environment']['nr_objectives']
        self.max_generations = parameters['environment']['max_generations']

        self.alternative_objectives = parameters['environment']['alternative_objectives']
        self.jobShopEnv = None

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.prev_embedding = 'none'
        self.reference_point = None
        self.generation = 0
        self.done = False
        self.shared_memory = None

        with open(BASE_PATH + '/datasets/ideal_points_{}_obj.json'.format(self.nr_objectives), 'r') as json_file:
            self.ideal_points = json.load(json_file)

        self.save_results = False
        if 'results_saving' in parameters:
            self.save_results = parameters['results_saving']['save_result']
            self.folder = parameters['results_saving']['folder']
            self.exp_name = parameters['results_saving']['exp_name']

        if 'problem_instances' in parameters['environment']:
            self.problem_instances = parameters['environment']['problem_instances']
        else:
            self.problem_instances = get_training_instances(parameters['environment']['problem_size'], parameters['environment']['instance_range'])

        self.bounds = []
        self.node_tracker = {}
        self.node_ids = {}
        self.graph_edges = []
        self.graph_edge_links = []


    def _observe(self):
        # Retrieve the current front and update the front tracker
        new_front = get_front(self)
        for ind in new_front:
            if ind.id not in self.node_ids:
                self.node_ids[ind.id] = len(self.node_ids.keys())
            if ind.id not in self.node_tracker:
                self.node_tracker[ind.id] = {"node_id": self.node_ids[ind.id], "fit": ind.fitness.values,
                                             "parents": ind.parents, "gen": self.generation}

        # Add self-loops for the current generation (each node connects to itself)
        if self.generation == 0:
            for id, individual_data in self.node_tracker.items():
                if id in [ind.id for ind in new_front]:  # Only add self-loops for nodes in the new front
                    edge_metadata = {"generation": self.generation}  # Generation info for the self-loop
                    node_id = self.node_ids[id]  # Unique ID for the node
                    edge_link = [node_id, node_id]  # Self-loop (source and target are the same)

                    # Check if the edge already exists
                    if edge_metadata not in self.graph_edges or edge_link not in self.graph_edge_links:
                        self.graph_edges.append(edge_metadata)  # Append edge attributes (self-loop)
                        self.graph_edge_links.append(edge_link)  # Append edge connection (self-loop)

        # Loop through the front tracker to create edges and edge links
        if self.generation > 0:  # Edges connect nodes between consecutive generations
            for ind in new_front:
                individual_data = self.node_tracker[ind.id]
                if individual_data["parents"] is not None:
                    for parent_id in individual_data["parents"]:
                        # Ensure the parent node is present in the graph_nodes
                        if parent_id in self.node_ids:
                            # Define edge attributes (generation and parent ID as attributes)
                            edge_metadata = {
                                "generation": ind.id[0] / self.max_generations,  # Generation info for the edge
                            }
                            edge_link = [self.node_ids[parent_id], self.node_ids[ind.id]]  # Connect parent to offspring
                            # Check if the edge already exists
                            if edge_metadata not in self.graph_edges or edge_link not in self.graph_edge_links:
                                self.graph_edges.append(edge_metadata)  # Append edge attributes
                                self.graph_edge_links.append(edge_link)  # Append edge connection

        # Convert the edge links and edges to the required array formats
        population_ids = [ind.id for ind in self.population]

        graph_nodes = []
        for node_id in self.node_tracker.keys():
            if node_id in population_ids:
                graph_nodes.append(self.node_tracker[node_id]['fit'] + (1.0,))
            else:
                graph_nodes.append(self.node_tracker[node_id]['fit'] + (0.0,))
        node_generations = [self.node_tracker[node]['gen'] for node in self.node_tracker.keys()]

        graph_nodes = np.array(normalize_graph_nodes(graph_nodes, self.bounds))
        previous_embedding = read_from_tensor_file(self.shared_memory)

        if previous_embedding.size != 0:
            if graph_nodes.shape[0] > previous_embedding.shape[0]:
                additional_zeros = torch.zeros(graph_nodes.shape[0]-previous_embedding.shape[0], 64) # TODO: hardcoded
                previous_embedding = torch.cat((previous_embedding, additional_zeros), dim=0)
        else:
            previous_embedding = 'none'

        # Create a dictionary with raw data for the observation
        obs = {
            "graph_nodes": graph_nodes,
            "node_generations": np.array(node_generations), # only used for coloring the nodes atm
            "graph_edges": np.array([edge["generation"] for edge in self.graph_edges]),  # Edge attributes (generation)
            "graph_edge_links": np.array(self.graph_edge_links).T,  # Edge connections (source, target)
            "prev_graph_embedding": previous_embedding,
            "shared_memory": self.shared_memory,
        }
        return [obs]

    def reset(self):
        self.prev_embedding = 'none'
        self.problem_instance = random.choice(self.problem_instances)

        self.jobShopEnv = load_job_shop_env(self.problem_instance)
        self.toolbox = base.Toolbox()
        # print('resetting')
        self.shared_memory = create_tensor_file(id(self))

        self.node_tracker = {}
        self.bounds = []
        self.node_ids = {}
        self.graph_edges = []
        self.graph_edge_links = []

        if not hasattr(creator, "Fitness"):
            creator.create("Fitness", base.Fitness, weights=tuple([-1.0 for i in range(self.nr_objectives)]))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.Fitness)

        self.toolbox.register("init_individual", init_individual, creator.Individual, jobShopEnv=self.jobShopEnv)
        self.toolbox.register("mate_TwoPoint", tools.cxTwoPoint)
        self.toolbox.register("mate_Uniform", tools.cxUniform, indpb=0.5)
        self.toolbox.register("mate_POX", pox_crossover, nr_preserving_jobs=1)

        self.toolbox.register("mutate_machine_selection", mutate_shortest_proc_time, jobShopEnv=self.jobShopEnv)
        self.toolbox.register("mutate_operation_sequence", mutate_sequence_exchange)
        self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.register("evaluate_individual", evaluate_individual, jobShopEnv=self.jobShopEnv,
                         objectives=self.nr_objectives)

        self.population = init_population(self.toolbox, self.population_size, )

        fitnesses = list(evaluate_population(self.toolbox, self.population, self.nr_objectives, None))
        for count, (ind, fit) in enumerate(zip(self.population, fitnesses)):
            ind.fitness.values = tuple(fit)  # Set the fitness values
            ind.id = (self.generation, count)
            ind.parents = None

        if self.save_results:
            self.hof = tools.ParetoFront()
            self.hof.update(self.population)

        self.bounds = get_fitness_bounds([ind.fitness.values for ind in self.population])
        if not self.save_results:
            self.reference_point = get_fitness_bounds([ind.fitness.values for ind in self.population])[1]
        else:
            if os.path.isfile(REFERENCE_POINTS_FILE):
                with open(REFERENCE_POINTS_FILE, 'r') as file:
                    reference_points = json.load(file)
                    if self.problem_instance in reference_points:
                        self.reference_point = reference_points[self.problem_instance][0:self.nr_objectives]
                        print('using reference point from file', self.reference_point)
                    else:
                        print('NO REFERENCE POINT KNOWN')

        self.generation = 0
        self.done = False
        self.initial_hv = self.step_hv = compute_hypervolume(self.population, self.nr_objectives, self.reference_point)

        if not self.save_results:
            self.ideal_hv = compute_hypervolume([tuple(self.ideal_points[self.problem_instance][:self.nr_objectives])], self.nr_objectives, self.reference_point)
        self.best_hv = self.initial_hv

        # Return the initial state
        return self._observe(), dict()

    def step(self, action):
        try:
            action1 = np.nan_to_num(action[0])
        except:
            action1 = action[0]
        action1 = np.clip(action1, -1, 1)

        try:
            action2 = np.nan_to_num(action[1])
        except:
            action2 = action[1]
        action2 = np.clip(action2, -1, 1)

        reward = 0
        self.generation += 1
        cxpb = (action1 + 1) * 0.2 + 0.6  # (between 0.6 and 1)
        mutpb = (action2 + 1) * 0.05  # (between 0 and 0.1)

        # Select the next generation
        offspring = []
        for _ in range(int(self.population_size)):
            if random.random() < cxpb:

                ind1, ind2 = list(map(self.toolbox.clone, random.sample(self.population, 2)))
                del ind1.parents
                if random.random() < 0.5:
                    if 'allocation' in ind1:
                        ind1['allocation'], ind2['allocation'] = self.toolbox.mate_TwoPoint(ind1['allocation'],
                                                                                       ind2['allocation'])
                    else:
                        ind1, ind2 = self.toolbox.mate_Uniform(ind1, ind2)

                else:
                    if 'allocation' in ind1:
                        ind1['allocation'], ind2['allocation'] = self.toolbox.mate_Uniform(ind1['allocation'],
                                                                                      ind2['allocation'])
                    else:
                        ind1[0], ind2[0] = self.toolbox.mate_Uniform(ind1[0], ind2[0])

                if 'order' in ind1:
                    ind1['order'], ind2['order'] = self.toolbox.mate_POX(ind1['order'], ind2['order'])
                else:
                    ind1[1], ind2[1] = self.toolbox.mate_POX(ind1[1], ind2[1])
                parent_ids = [ind1.id, ind2.id]
                ind1.parents = parent_ids
            else:
                ind1 = self.toolbox.clone(random.choice(self.population))
                del ind1.parents
                parent_ids = [ind1.id]
                ind1.parents = parent_ids

            # Apply mutation
            if 'allocation' in ind1:
                ind1['allocation'] = self.toolbox.mutate_machine_selection(ind1['allocation'], mutpb)
                ind1['order'] = self.toolbox.mutate_operation_sequence(ind1['order'], mutpb)
            else:
                ind1[0] = self.toolbox.mutate_machine_selection(ind1[0], mutpb)
                ind1[1] = self.toolbox.mutate_operation_sequence(ind1[1], mutpb)

            del ind1.fitness.values
            del ind1.id
            offspring.append(ind1)

        if '/dafjs/' in self.problem_instance or '/yfjs/' in self.problem_instance:
            offspring = repair_precedence_constraints(self.jobShopEnv, offspring)

        # Evaluate the population
        # sequential evaluation of population
        fitnesses = list(evaluate_population(self.toolbox, offspring, self.nr_objectives, None))
        for count, (ind, fit) in enumerate(zip(offspring, fitnesses)):
            ind.fitness.values = tuple(fit)  # Set the fitness values
            ind.id = (self.generation, count)
            #ind.parents = None

        if self.save_results:
            self.hof.update(offspring)

        self.bounds = update_bounds(self.bounds, fitnesses)

        # Select next generation population
        self.population = self.toolbox.select(self.population + offspring, self.population_size)

        if not self.save_results:
            episode_hv = compute_hypervolume(self.population, self.nr_objectives, self.reference_point)
            # reward = ((episode_hv - self.initial_hv) / (self.ideal_hv - self.initial_hv)) * 100

            if self.best_hv < episode_hv:
                current_gap = (episode_hv - self.initial_hv) / (self.ideal_hv - self.initial_hv) * 100
                previous_gap = (self.best_hv - self.initial_hv) / (self.ideal_hv - self.initial_hv) * 100
                reward = round(((current_gap) ** 2) - ((previous_gap) ** 2), 1)
                self.best_hv = episode_hv

        if self.generation == self.max_generations:
            self.done = True
            if self.save_results:
                print('Saving results')
                self.save_result()

        # Return new state, reward, done, and optional info
        return self._observe(), reward, self.done, None, {}

    # --------------------------------------------------------------------------------------------------------------------
    def sample(self, nr_episodes=5, plotting=False, plot_every=10):
        """
        Sample random actions and run the environment
        """
        for episode in range(nr_episodes):
            #start_time = time.time()

            print("Start episode:", episode)
            obs, _ = self.reset()
            if plotting:
                print('plotting at generation', self.generation)
                self.plot_observation(obs)
            while True:
                action = self.action_space.sample()
                obs, reward, done, _, _ = self.step(action)
                if plotting and self.generation % plot_every == 0:
                    print('plotting at generation', self.generation)
                    self.plot_observation(obs)
                if done:
                    #end_time = time.time()  # End time for the episode
                    #duration = end_time - start_time  # Calculate the duration
                    #print(f"Episode {episode} completed in {duration:.2f} seconds")
                    break

    def plot_observation(self, obs):
        """
        Plot the observation with edges connecting nodes based on parent relationships.
        The x and y coordinates represent objectives, and node color indicates generation.
        """

        graph_nodes = obs[0]["graph_nodes"]  # Node fitness values (objectives)
        edges = obs[0]["graph_edges"]  # Edge attributes (generation)
        edge_links = obs[0]["graph_edge_links"]  # Edge connections (source, target)

        # Create a directed graph
        graph = nx.DiGraph()
        for i, node in enumerate(graph_nodes):
            graph.add_node(i, fitness=node)  # Add nodes with fitness as an attribute

        # # Add edges based on edge_links
        # for i in range(edge_links.shape[1]):
        #     source, target = edge_links[:, i]
        #     generation = edges[i]
        #     graph.add_edge(source, target, generation=generation)

        # Add edges, skipping self-loops
        # Add edges, skipping self-loops and duplicate-coordinate links
        for i in range(edge_links.shape[1]):
            source, target = edge_links[:, i]

            # Skip self-loop
            if source == target:
                continue

            # Skip edge if coordinates are the same
            if graph_nodes[source][0] == graph_nodes[target][0] and graph_nodes[source][1] == graph_nodes[target][1]:
                continue

            generation = edges[i]
            graph.add_edge(source, target, generation=generation)

        # Extract x and y coordinates from objectives
        x_coords = [node[0] for node in graph_nodes]  # First objective
        y_coords = [node[1] for node in graph_nodes]  # Second objective
        in_population = [int(node[self.nr_objectives]) for node in graph_nodes]  # In population
        node_colors = list(obs[0]['node_generations'])
        edge_colors = ['black' if in_population[i] == 0 else 'red' for i in range(len(in_population))]

        # # Plot nodes and edges
        # plt.figure(figsize=(10, 8))
        # cmap = plt.cm.plasma  # Use a color map for generation
        # scatter = plt.scatter(x_coords, y_coords, c=node_colors, cmap=cmap, s=500, edgecolors=edge_colors, label="Nodes")
        #
        # # Add edges
        # for source, target in graph.edges:
        #     plt.plot(
        #         [x_coords[source], x_coords[target]],
        #         [y_coords[source], y_coords[target]],
        #         "k-", alpha=0.5
        #     )

        # Create position dictionary for graph
        pos = {i: (x_coords[i], y_coords[i]) for i in range(len(graph_nodes)) if x_coords[i] != y_coords[i]}

        # Plot nodes and edges

        plt.figure(figsize=(12, 10))
        cmap = plt.cm.plasma  # Use a color map for generation
        scatter = plt.scatter(x_coords, y_coords, c=node_colors, cmap=cmap, s=500, edgecolors=edge_colors,
                              label="Nodes")

        # Draw directed edges with arrows
        nx.draw_networkx_edges(
            graph,
            pos,
            arrows=True,
            arrowstyle='->',
            arrowsize=35,
            edge_color='#696969',
            alpha=0.9
        )

        # Add color bar for generation
        cbar = plt.colorbar(scatter)
        cbar.set_label("Generation", rotation=270, labelpad=30, fontsize=26)
        cbar.ax.xaxis.label.set_horizontalalignment('center')
        cbar.ax.tick_params(labelsize=30)  # Change tick label size for the colorbar

        plt.xlabel("Objective 1", fontsize=26)
        plt.ylabel("Objective 2", fontsize=26)
        #plt.title("Pareto Front Connectivity with Parent Relationships", fontsize=16)

        # plt.xticks(fontsize=24)  # Change x-axis tick label size
        # plt.yticks(fontsize=24)  # Change y-axis tick label size

        plt.grid(True)
        #plt.grid(False)

        plt.tick_params(axis='x', which='both', labelbottom=True)  # Enable x-axis tick labels
        plt.tick_params(axis='y', which='both', labelleft=True)  # Enable y-axis tick labels

        # # Hide tick labels
        # plt.xticks(color='none')
        # plt.yticks(color='none')
        #
        # # Hide tick marks
        # plt.tick_params(axis='x', which='both', bottom=False, top=False)
        # plt.tick_params(axis='y', which='both', left=False, right=False)

        # Hide the top and right spines
        #plt.savefig('plot_transparent.png', format='png', dpi=600, transparent=True)
        plt.show()

    def save_result(self):
        output_dir = self.folder
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Ensure that exp_name includes a slash (/) if needed
        exp_name = self.exp_name.strip("/")

        results = {}
        results['problem_instance'] = self.problem_instance
        results['hypervolume'] = compute_hypervolume(self.hof, self.nr_objectives, self.reference_point)

        # #REBUTTAL: ADDITIONAL OBJECTIVES
        # from pymoo.indicators.igd import IGD
        # from pymoo.indicators.igd_plus import IGDPlus
        # first_front = tools.sortNondominated(self.hof, len(self.hof), first_front_only=True)[0]
        # results['nr_of_solutions'] = len(first_front)
        #
        # if self.nr_objectives == 2:
        #     with open('/hpc/za64617/projects/GNN4APC_dev/code/src/scheduling/data/approximated_fronts_2_obj.json',
        #               'r') as file:
        #         approximated_fronts = json.load(file)
        #
        # elif self.nr_objectives == 3:
        #     with open('/hpc/za64617/projects/GNN4APC_dev/code/src/scheduling/data/approximated_fronts_3_obj.json',
        #               'r') as file:
        #         approximated_fronts = json.load(file)
        #
        # elif self.nr_objectives == 5:
        #     with open('/hpc/za64617/projects/GNN4APC_dev/code/src/scheduling/data/approximated_fronts_5_obj.json',
        #               'r') as file:
        #         approximated_fronts = json.load(file)
        #
        # approximated_front = approximated_fronts[self.problem_instance]
        # igd = IGD(np.array(approximated_front))
        # results['igd'] = igd.do(np.array([list(item.fitness.values) for item in self.hof]))
        # igd_plus = IGDPlus(np.array(approximated_front))
        # results['igd_plus'] = igd_plus.do(np.array([list(item.fitness.values) for item in self.hof]))

        results_csv_path = os.path.join(output_dir, f'{exp_name}_results.csv')
        df = pd.DataFrame.from_dict(results, orient='index').T
        pd.DataFrame(df).to_csv(results_csv_path, index=False)


# --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    config_filepath = BASE_PATH + "STN_MODAC/configs/config_scheduling_fjsp.toml"
    with open(config_filepath, 'r') as toml_file:
        config = toml.load(toml_file)
    env = schedulingEnv(config) #, EmbeddingTracker(config['environment']['population_size']))
    env.sample(plotting=True)