from torch_geometric.nn import GCNConv
from torch.nn import BatchNorm1d as BatchNorm
from gnn.gnn import GNN


class GCN(GNN):
    def __init__(self, nfeat, nhid, nclass, num_layers, dropout=0.5, **kwargs):
        super().__init__(nfeat, nhid, nclass, num_layers, dropout, pooling='mean')

        for _ in range(num_layers):
            conv = GCNConv(nfeat, nhid)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(nhid))
            nfeat = nhid
