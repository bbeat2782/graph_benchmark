import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, nfeat, nclass_node, nclass_graph, nhid=64, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid)
        self.dropout = dropout

        self.node_classifier = nn.Linear(nhid, nclass_node)
        self.graph_classifier = nn.Linear(nhid, nclass_graph)

    def forward(self, data, task='node'):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        if task == 'node':
            x = F.dropout(x, self.dropout, training=self.training)
            return F.log_softmax(self.node_classifier(x), dim=1)
        elif task == 'graph':
            batch = data.batch
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = global_mean_pool(x, batch)
            x = F.dropout(x, self.dropout, training=self.training)
            return F.log_softmax(self.graph_classifier(x), dim=1)
        else:
            raise ValueError(f"Task must be 'node' or 'graph'. Current task value: {task}")
