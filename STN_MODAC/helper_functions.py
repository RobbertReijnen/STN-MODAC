import numpy as np
from deap import tools


def get_front(env):
    # Perform non-dominated sorting on the population
    fronts = tools.sortNondominated(env.population, len(env.population), first_front_only=False)[0]

    return fronts


def get_fitness_bounds(fitnesses):
    fitnesses = np.array(fitnesses)

    # Calculate min and max bounds
    min_bounds = np.min(fitnesses, axis=0)
    max_bounds = np.max(fitnesses, axis=0)

    # Calculate bounds
    bounds = np.array([min_bounds, max_bounds])
    return bounds


def update_bounds(bounds, fitness):
    fitness = np.array(fitness)

    # Calculate min and max bounds for each objective separately
    min_bounds = np.min(fitness, axis=0)
    max_bounds = np.max(fitness, axis=0)

    # Update bounds for each objective separately
    for i in range(len(min_bounds)):
        bounds[0][i] = np.minimum(bounds[0][i], min_bounds[i])
        bounds[1][i] = np.maximum(bounds[1][i], max_bounds[i])

    return bounds


def normalize_graph_nodes(graph_nodes, bounds):
    try:
        # Ensure graph_nodes is a 2D numpy array
        graph_nodes = np.array(graph_nodes)

        # Check if graph_nodes has more than 1 dimension
        if graph_nodes.ndim == 1:
            # If it's a 1D array (i.e., a single node), convert it into a 2D array
            graph_nodes = graph_nodes.reshape(1, -1)

        # Ensure we normalize only the first two columns
        num_columns_to_normalize = min(graph_nodes.shape[1], bounds.shape[1])

        # Normalize each relevant objective (column) of the graph nodes
        normalized_nodes = np.zeros_like(graph_nodes)

        for i in range(num_columns_to_normalize):  # For each relevant objective (column)
            min_bound = bounds[0, i]
            max_bound = bounds[1, i]

            # Handle edge case where max_bound == min_bound
            if max_bound == min_bound:
                print(f"Column {i}: max_bound equals min_bound ({min_bound}). Assigning constant value 0.")
                normalized_nodes[:, i] = 0  # Assign a constant value if bounds are the same
            else:
                # Apply min-max normalization
                normalized_nodes[:, i] = (graph_nodes[:, i] - min_bound) / (max_bound - min_bound)

        # Retain non-normalized values for extra columns
        if graph_nodes.shape[1] > num_columns_to_normalize:
            normalized_nodes[:, num_columns_to_normalize:] = graph_nodes[:, num_columns_to_normalize:]

        # Validate the output for NaN or inf values
        if np.isnan(normalized_nodes).any() or np.isinf(normalized_nodes).any():
            raise ValueError("Invalid values found in normalized_nodes after normalization.")

        return normalized_nodes

    except Exception as e:
        print("Error during normalization:")
        print(f"Graph Nodes: {graph_nodes}")
        print(f"Bounds: {bounds}")
        print(f"Exception: {e}")
        raise  # Re-raise the exception after logging for debugging