import torch
from torch import nn
from torch_geometric.nn import GINConv, global_add_pool
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch.nn import BatchNorm1d as BatchNorm


class GIN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

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

        self.lin1 = Linear(nhid, nhid)
        self.batch_norm1 = BatchNorm(nhid)
        self.lin2 = Linear(nhid, nclass)

    def forward(self, data, task='node'):
        x, edge_index = data.x, data.edge_index
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))

        if task == 'graph':
            batch = data.batch
            x = global_add_pool(x, batch)
        x = F.relu(self.batch_norm1(self.lin1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=1)
