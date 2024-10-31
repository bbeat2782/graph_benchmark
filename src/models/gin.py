import torch
from torch import nn
from torch_geometric.nn import GINConv
from torch.nn import Sequential, Linear, ReLU
from torch.nn import BatchNorm1d as BatchNorm
from .gnn import GNN


class GIN(GNN):
    def __init__(self, nfeat, nhid, nclass, num_layers, dropout=0.5, **kwargs):
        super().__init__(nfeat, nhid, nclass, num_layers, dropout, pooling='add')

        for _ in range(num_layers):
            mlp = Sequential(
                Linear(nfeat, 2 * nhid),
                BatchNorm(2 * nhid),
                ReLU(),
                Linear(2 * nhid, nhid)
            )
            conv = GINConv(mlp)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(nhid))
            nfeat = nhid
