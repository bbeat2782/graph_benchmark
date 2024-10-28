from torch_geometric.nn import GATConv
from torch.nn import BatchNorm1d as BatchNorm
from gnn.gnn import GNN


class GAT(GNN):
    def __init__(self, nfeat, nhid, nclass, num_layers, heads=1, dropout=0.5, **kwargs):
        super().__init__(nfeat, nhid, nclass, num_layers, dropout, pooling='add', heads=heads)

        for _ in range(num_layers):
            conv = GATConv(nfeat, nhid, heads=heads, concat=True)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(nhid * heads))
            nfeat = nhid * heads
