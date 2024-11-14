import sys
import os
import json
import torch
import yaml
from src.utils.utils import Train, plot_accuracy, changing_num_layers, plot_boxplot
from src.models.gcn import GCN
from src.models.gin import GIN
from src.models.gat import GAT

from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
from argparse import Namespace
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric import seed_everything
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from graphgps.logger import create_logger


# TODO
# 1. code for testing multiple layers
# 2. Check whether random split for gps-peptides is working as expected
# 3. make a separate function that saves a plot using result.json
# 4. make use of graph_pooling, dropout, clip_grad_norm, weight_decay
# 5. clean earlystopping

def custom_set_out_dir(cfg, cfg_fname, name_tag):
    """Set custom main output directory path to cfg.
    Include the config filename and name_tag in the new :obj:`cfg.out_dir`.

    Args:
        cfg (CfgNode): Configuration node
        cfg_fname (string): Filename for the yaml format configuration file
        name_tag (string): Additional name tag to identify this execution of the
            configuration file, specified in :obj:`cfg.name_tag`
    """
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)


def custom_set_run_dir(cfg, run_id):
    """Custom output directory naming for each experiment run.

    Args:
        cfg (CfgNode): Configuration node
        run_id (int): Main for-loop iter id (the random seed or dataset split)
    """
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)


def main():
    model_dataset_to_evaluate = [
        'configs/GraphGPS/GPS_peptides-func.yaml',
        # 'configs/GCN/GCN_cora.yaml',
        'configs/GIN/GIN_cora.yaml',
        # 'configs/GAT/GAT_cora.yaml',
        # 'configs/GCN/GCN_enzymes.yaml',
        # 'configs/GIN/GIN_enzymes.yaml',
        'configs/GAT/GAT_enzymes.yaml',
        'configs/GCN/GCN_imdb_binary.yaml',
        # 'configs/GIN/GIN_imdb_binary.yaml',
        # 'configs/GAT/GAT_imdb_binary.yaml',
        # 'configs/GCN/GCN_peptides-func.yaml',
        # 'configs/GIN/GIN_peptides-func.yaml',
        # 'configs/GAT/GAT_peptides-func.yaml',
    ]

    result = {}
    for config_file in model_dataset_to_evaluate:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        dataset_name = config['dataset']['name']
        model_name = config['model']['type']
        metric = config['metric_best']
        print(f'Training and evaluating {model_name} on {dataset_name}...')

        if dataset_name not in result:
            result[dataset_name] = {}

        if model_name == 'GPSModel':
            set_cfg(cfg)
            cfg.set_new_allowed(True)
            args = Namespace(cfg_file=config_file, opts=[])
            load_cfg(cfg, args)
            if not hasattr(cfg, 'name_tag'):
                cfg.name_tag = "default_tag"
            custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
            dump_cfg(cfg)

            torch.set_num_threads(cfg.num_threads)
            custom_set_run_dir(cfg, 0)
            set_printing()
            cfg.seed = 42
            cfg.run_id = 0
            seed_everything(cfg.seed)
            auto_select_device()

            loaders = create_loader()
            loggers = create_logger()

            trainer = Train(config, loaders)
        else:
            trainer = Train(config)
        # TODO need an extra function to get the best metric using val_loss
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
