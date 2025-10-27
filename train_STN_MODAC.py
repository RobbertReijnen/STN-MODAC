import os
import torch
import helper_functions_main as helper_functions
import pprint
import numpy as np
import tianshou as ts

from tianshou.utils import TensorboardLogger
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils.net.common import ActorCritic
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Independent, Normal
from torch_geometric.nn import GCNConv
from torch.optim.lr_scheduler import LambdaLR

from STN_MODAC.environment_scheduling import schedulingEnv
from STN_MODAC.gnn_models import GNNActor, GNNCritic, GraphEncoder

from config import BASE_PATH

METHOD = 'STN_MODAC'
TRAIN_CONFIG_FILE = BASE_PATH + METHOD + '/configs/config_scheduling_fjsp.toml'


def initialize_actor_critic(actor_critic):
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, GCNConv):
            if hasattr(m, 'lin'):
                torch.nn.init.orthogonal_(m.lin.weight)
                if m.lin.bias is not None:
                    torch.nn.init.zeros_(m.lin.bias)

    # do last policy layer scaling, this will make initial actions have (close to)
    # 0 mean and std, and will help boost performances,
    # see https://arxiv.org/abs/2006.05990, Fig.24 for details
    actor = actor_critic.actor
    for m in actor.fc_mu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)


def main():
    config = helper_functions.load_config(TRAIN_CONFIG_FILE)
    device = helper_functions.setup_device(config)
    model_logdir = helper_functions.create_log_directory(METHOD, config, TRAIN_CONFIG_FILE)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), model_logdir + '/best_model/policy.pth')

    def save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int):
        if epoch % 100 == 0:
            ckpt_path = os.path.join(model_logdir + '/intermediate_models/', f"policy_epoch_{epoch}.pth")
            torch.save(policy.state_dict(), ckpt_path)

    writer = SummaryWriter(log_dir=model_logdir)

    if config['environment']['nr_of_environments'] == 1:
        train_env = DummyVectorEnv(
            [lambda: schedulingEnv(config)
             for _ in range(config['environment']['nr_of_environments'])])

    elif config['environment']['nr_of_environments'] >= 1:
        train_env = SubprocVectorEnv(
            [lambda: schedulingEnv(config)
             for _ in range(config['environment']['nr_of_environments'])])

    graph_encoder = GraphEncoder(input_dim=config['environment']['nr_objectives']+1,
                                 hidden_dim=config['policy']['actor_hidden_dim'])
    actor = GNNActor(hidden_dim=config['policy']['actor_hidden_dim'],
                     action_shape=train_env.action_space[0],
                     device=device,
                     graph_encoder=graph_encoder).to(device)
    critic = GNNCritic(hidden_dim=config['policy']['critic_hidden_dim'],
                       device=device,
                       graph_encoder=graph_encoder).to(device)
    critic.share_memory()

    actor_critic = ActorCritic(actor, critic)
    initialize_actor_critic(actor_critic)

    optim = torch.optim.Adam(actor_critic.parameters(), lr=config['ppo']['learning_rate'])
    lr_scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 1 - epoch / config['ppo']['max_epoch']) if config['ppo']['lr_decay'] else None

    policy = ts.policy.PPOPolicy(actor=actor,
                                 critic=critic,
                                 optim=optim,
                                 dist_fn=lambda *logits: Independent(Normal(*logits), 1),
                                 discount_factor=config['ppo']['gamma'],
                                 gae_lambda=config['ppo']['gae_lambda'],
                                 max_grad_norm=config['ppo']['max_grad_norm'],
                                 vf_coef=config['ppo']['vf_coef'],
                                 ent_coef=config['ppo']['ent_coef'],
                                 reward_normalization=config['ppo']['reward_normalization'],
                                 action_scaling=True,
                                 action_bound_method=config['ppo']['action_bound_method'],
                                 lr_scheduler=lr_scheduler,
                                 action_space=config['environment']['nr_actions'],
                                 eps_clip=config['ppo']['eps_clip'],
                                 value_clip=config['ppo']['value_clip'],
                                 dual_clip=None,
                                 advantage_normalization=config['ppo']['advantage_normalization'],
                                 recompute_advantage=config['ppo']['recompute_advantage'])

    buffer = VectorReplayBuffer(config['ppo']['buffer_size'], config['environment']['nr_of_environments'])
    train_collector = Collector(policy, train_env, buffer, preprocess_fn=None, exploration_noise=True)
    logger = TensorboardLogger(writer)

    result = ts.trainer.onpolicy_trainer(policy, train_collector, max_epoch=config['ppo']['max_epoch'],
                                         step_per_epoch=config['ppo']['step_per_epoch'],
                                         episode_per_collect=config['ppo']['episode_per_collect'],
                                         save_checkpoint_fn=save_checkpoint_fn,
                                         save_best_fn=save_best_fn,
                                         batch_size=config['ppo']['batch_size'], test_collector=None,
                                         repeat_per_collect=1, episode_per_test=0, logger=logger)
    pprint.pprint(result)
    torch.save(policy.state_dict(), os.path.join(model_logdir, 'policy.pth'))


if __name__ == '__main__':
    main()