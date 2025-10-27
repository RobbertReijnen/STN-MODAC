import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_geometric_temporal.nn.recurrent import TGCN
from STN_MODAC.shared_memory import write_to_tensor_file


class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.tgcn = TGCN(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr, batch, hidden_state):
        hidden_state = self.tgcn(x, edge_index, edge_attr, hidden_state)
        output = global_mean_pool(hidden_state, batch=batch)
        return self.linear(output), hidden_state


def prepare_data(observations, device):
    data_list = []
    for observation in observations:
        graph_nodes = torch.tensor(observation[0]["graph_nodes"], dtype=torch.float32, device=device)
        graph_edge_links = torch.tensor(observation[0]["graph_edge_links"], dtype=torch.long, device=device)
        graph_edges = torch.tensor(observation[0]["graph_edges"], dtype=torch.float32, device=device)
        kwargs = {"prev_graph_embedding" : observation[0]["prev_graph_embedding"],
                  "shared_memory": observation[0]['shared_memory']}
        data = Data(x=graph_nodes, edge_index=graph_edge_links, edge_attr=graph_edges, **kwargs)
        data_list.append(data)
    return data_list


class GNNActor(nn.Module):
    def __init__(self, hidden_dim, action_shape, device, graph_encoder):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.hidden_dim = hidden_dim
        self.output_dim = int(action_shape.shape[0])
        self.additional_feature_vector_dim = 0
        self.device = device

        self.fc_mu = nn.Linear(hidden_dim + self.additional_feature_vector_dim, self.output_dim)
        self.sigma_param = nn.Parameter(torch.zeros(self.output_dim, 1))

    def forward(self, observations, state=None, info={}):
        data_list = prepare_data(observations, self.device)

        mu_list = []
        sigma_list = []

        for data in data_list:
            x = data.x.float()
            edge_index = data.edge_index.long()
            edge_attr = data.edge_attr.float()
            hidden_state = data.get('prev_graph_embedding')

            if hidden_state == "niks":
                hidden_state = None
            batch = data.batch

            if hidden_state is None:
                print('forwarding: x:', x.shape, 'hidden_state:', hidden_state)
            else:
                print('forwarding: x:', x.shape, 'hidden_state:', hidden_state.shape)
            x, hidden_state = self.graph_encoder(x, edge_index, edge_attr, batch, hidden_state)

            # Process hidden state for the graph
            shared_memory = data.get('shared_memory')
            write_to_tensor_file(shared_memory, hidden_state)

            mu = self.fc_mu(x)
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()

            mu_list.append(mu)
            sigma_list.append(sigma)

        # Combine results from all graphs in the batch
        mu = torch.cat(mu_list, dim=0)
        sigma = torch.cat(sigma_list, dim=0)

        return (mu, sigma), state


class GNNCritic(nn.Module):
    def __init__(self, hidden_dim, device, graph_encoder):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.graph_encoder = graph_encoder
        self.additional_feature_vector_dim = 0
        self.fc = nn.Linear(hidden_dim + self.additional_feature_vector_dim, 1)
        self.device = device

    def forward(self, observations, state=None, info={}):
        data_list = prepare_data(observations, self.device)

        # Now we iterate over the data_list, processing each graph separately
        value_list = []

        for data in data_list:
            x = data.x.float()
            edge_index = data.edge_index.long()
            edge_attr = data.edge_attr.float()
            hidden_state = data.get('prev_graph_embedding', None)
            # print('hidden_state gnn critic:', hidden_state)
            if hidden_state == "none":
                hidden_state = None
            batch = data.batch

            x, hidden_state = self.graph_encoder(x, edge_index, edge_attr, batch, hidden_state)

            # Process hidden state for the graph
            value = self.fc(x)
            value_list.append(value)

        # Combine results from all graphs in the batch
        value = torch.cat(value_list, dim=0)

        return value