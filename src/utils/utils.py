import os
import torch
import numpy as np
import random
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def setup_seed(seed=42):
    """
    Set seed for reproducibility
    """
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def add_node_features(data):
    """
    For dataset that has no node feature,
    add node degree as node feature for each node
    """
    G = to_networkx(data, to_undirected=data.is_undirected())
    degrees = np.array([G.degree(n) for n in range(G.number_of_nodes())], dtype=np.float32)
    features = torch.tensor(degrees).view(-1, 1)

    return Data(x=features, edge_index=data.edge_index, y=data.y)
