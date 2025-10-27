import torch
import os
import helper_functions_main as helper_functions
import argparse

from tianshou.env import DummyVectorEnv
from torch.distributions import Independent, Normal
import tianshou as ts

from STN_MODAC.environment_scheduling import schedulingEnv
from STN_MODAC.gnn_models import GNNActor, GNNCritic, GraphEncoder
from scheduling.helper_functions import load_parameters

PARAM_FILE = "configs/STN_MODAC_GA.json"
DEFAULT_RESULTS_ROOT = "./results/single_runs_drl"


def run_algo(folder, exp_name, **exp_config):
    train_config_file = exp_config['model_path'][:exp_config['model_path'].rfind('/')] + '/train_config.toml'
    config = helper_functions.load_config(train_config_file)
    config['results_saving'] = {}
    config['results_saving']['folder'] = folder
    config['results_saving']['exp_name'] = exp_name
    config['results_saving']['save_result'] = True
    config['environment']['population_size'] = exp_config['population_size']
    config['environment']['max_generations'] = exp_config['ngen']
    config['environment']['problem_instances'] = exp_config['test_problem_instance']
    config['environment']['nr_objectives'] = exp_config['nr_of_objectives']

    test_env = DummyVectorEnv([lambda: schedulingEnv(config)])
    device = 'cpu' #utils.setup_device(config)

    graph_encoder = GraphEncoder(input_dim=config['environment']['nr_objectives']+1,
                                 hidden_dim=config['policy']['actor_hidden_dim'])
    actor = GNNActor(hidden_dim=config['policy']['actor_hidden_dim'],
                     action_shape=test_env.action_space[0],
                     device=device,
                     graph_encoder=graph_encoder).to(device)
    critic = GNNCritic(hidden_dim=config['policy']['critic_hidden_dim'],
                       device=device,
                       graph_encoder=graph_encoder).to(device)
    # Create the policy
    policy = ts.policy.PPOPolicy(actor=actor,
                                 critic=critic,
                                 optim=None,  # Not needed for testing
                                 dist_fn=lambda *logits: Independent(Normal(*logits), 1))


    policy.load_state_dict(torch.load(exp_config['model_path']))

    policy.eval()
    collector = ts.data.Collector(policy, test_env, exploration_noise=False)
    result = collector.collect(n_episode=1)


def main(param_file=PARAM_FILE):
    parameters = load_parameters(param_file)
    folder = DEFAULT_RESULTS_ROOT
    exp_name = 'test_run'
    run_algo(folder, exp_name, **parameters)


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