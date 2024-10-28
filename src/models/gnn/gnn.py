from torch.nn import Linear
from torch_geometric.nn import global_add_pool, global_mean_pool
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BatchNorm


class GNN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, num_layers, dropout=0.5, pooling='add', heads=1, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.pooling = pooling
        self.heads = heads

        # Define these in children classes
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        # Line for GAT
        final_nhid = nhid * heads if heads > 1 else nhid
        self.lin1 = Linear(final_nhid, nhid)
        self.batch_norm1 = BatchNorm(nhid)
        self.lin2 = Linear(nhid, nclass)

    def forward(self, data, task='node'):
        x, edge_index = data.x, data.edge_index

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))

        # Task-specific operations
        if task == 'graph':
            batch = data.batch
            if self.pooling == 'add':
                x = global_add_pool(x, batch)
            elif self.pooling == 'mean':
                x = global_mean_pool(x, batch)
            else:
                raise ValueError(f"Unknown pooling method: {self.pooling}")

        # Classifier layer
        x = F.relu(self.batch_norm1(self.lin1(x)))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=1)
