import sys
import os
import json
import yaml
from src.utils.utils import Train, plot_accuracy, changing_num_layers, default_test, plot_boxplot
from src.models.gcn import GCN
from src.models.gin import GIN
from src.models.gat import GAT

# TODO
# 1. Use one training function for Cora and peptides rather than separate ones
# 2. implement num_repetition to repeat training with different train/val/test splits
# 3. make .yaml files for Cora, Enzymes, and IMDB
# 4. run with Cora, Enzymes, and IMDB
# 5. make use of graph_pooling, dropout, clip_grad_norm, weight_decay, schedulr, num_warmup_epochs
# 6. make a separate function that saves a plot using result.json
# 7. Copy GraphGPS code??



def main():
    model_dataset_to_evaluate = [
        'configs/GCN/GCN_peptides-func.yaml',
        'configs/GIN/GIN_peptides-func.yaml',
        'configs/GAT/GAT_peptides-func.yaml',
    ]

    result = {}
    for config_file in model_dataset_to_evaluate:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        dataset_name = config['dataset']['name']
        model_name = config['model']['type']
        metric = config['metric_best']
        print(f'Training and evaluating {model_name} on {dataset_name}...')

        # Initialize the dataset entry
        if dataset_name not in result:
            result[dataset_name] = {}

        trainer = Train(config)
        train_losses, val_losses, test_losses, train_metrics, val_metrics, test_metrics = trainer()
        result[dataset_name][model_name] = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'test_losses': test_losses,
                f'train_{metric}s': train_metrics,
                f'val_{metric}s': val_metrics,
                f'test_{metric}s': test_metrics
            }


    output_file = "results/result_testing.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=4)
    
    print(f"Results saved to {output_file}")

    return None

        

    # # Experiment 1
    # result = {}
    # for dataset, task in zip(datasets, tasks):
    #     result[dataset] = {}
    #     for model, name in zip([GCN, GIN, GAT], ['GCN', 'GIN', 'GAT']):
    #         train_accs, val_accs, test_accs = default_test(dataset, task, model, num_iterations=30, num_layer=2, nhid=nhid, heads=heads, lr=lr, batch_size=batch_size, epochs=epochs)
    #         result[dataset][name] = {
    #             'train_accs': train_accs,
    #             'val_accs': val_accs,
    #             'test_accs': test_accs
    #         }

    # plot_boxplot(result)

    # # Experiment 2
    # # Testing how different GNNs perform when number of layers changes
    # result = {}
    # for dataset, task in zip(datasets, tasks):
    #     for model, name in zip([GCN, GIN, GAT], ['GCN', 'GIN', 'GAT']):
    #         train_accs, val_accs, test_accs = changing_num_layers(dataset, task, model, nhid=nhid, heads=heads, lr=lr, batch_size=batch_size, epochs=epochs)
    #         result[name] = {
    #             'train_accs': train_accs,
    #             'val_accs': val_accs,
    #             'test_accs': test_accs
    #         }

    #     plot_accuracy(result, dataset)


if __name__ == "__main__":
    main()
