import sys
import os
import json
from src.utils.utils import Train, plot_accuracy, changing_num_layers
from src.models.gcn import GCN
from src.models.gin import GIN
from src.models.gat import GAT


def main(nhid=64, heads=4, lr=0.001, batch_size=64, epochs=500, datasets=['Cora', 'ENZYMES', 'IMDB-BINARY'], tasks=['node', 'graph', 'graph']):

    # Testing how different GNNs perform when number of layers changes
    result = {}
    for dataset, task in zip(datasets, tasks):
        for model, name in zip([GCN, GIN, GAT], ['GCN', 'GIN', 'GAT']):
            train_accs, val_accs, test_accs = changing_num_layers(dataset, task, model, nhid=nhid, heads=heads, lr=lr, batch_size=batch_size, epochs=epochs)
            result[name] = {
                'train_accs': train_accs,
                'val_accs': val_accs,
                'test_accs': test_accs
            }

        plot_accuracy(result, dataset)


    return None


if __name__ == '__main__':
    # Default values
    nhid = 64
    heads = 4
    lr = 0.001
    batch_size = 64
    epochs = 500

    # Parse command-line arguments if provided
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            key, value = arg.split('=')
            if key == 'nhid':
                nhid = int(value)
            elif key == 'heads':
                heads = int(value)
            elif key == 'lr':
                lr = float(value)
            elif key == 'batch_size':
                batch_size = int(value)
            elif key == 'epochs':
                epochs = int(value)

    main(nhid=nhid, heads=heads, lr=lr, batch_size=batch_size, epochs=epochs)
