import os
import torch
import numpy as np
from torch import nn
import random
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.utils import to_networkx, degree
from torch_geometric.datasets import TUDataset, Planetoid, LRGBDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import torch.nn.functional as F
from .earlystopping import EarlyStoppingLoss
import pandas as pd
from sklearn.metrics import f1_score, average_precision_score, accuracy_score
from torch_geometric.transforms import BaseTransform
from ..models.gcn import GCN
from ..models.gin import GIN
from ..models.gat import GAT
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.utils import add_self_loops
from copy import deepcopy
from transformers import get_cosine_schedule_with_warmup



class CustomDataset(Dataset):
    def __init__(self, root, name, transform=None):
        super().__init__(root, transform)
        self.name = name
        self.raw_dataset = TUDataset(root=root, name=name)
        self.data_list = [add_node_features(self.raw_dataset[i]) for i in range(len(self.raw_dataset))]

        # Set nfeat and nclass based on the transformed data
        self.nfeat = self.data_list[0].x.shape[1]
        self.nclass = len(set(data.y.item() for data in self.data_list))

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


class ModifiedDataset(InMemoryDataset):
    def __init__(self, original_dataset):
        # Initialize the parent InMemoryDataset
        super().__init__(root=original_dataset.root)
        # Modify each graph in the original dataset
        modified_data_list = [self.modify_data(data) for data in original_dataset]
        # Collate the modified graphs into a single dataset object
        self.data, self.slices = self.collate(modified_data_list)

    def modify_data(self, data):
        # Add default edge attributes if missing
        if data.edge_attr is None:
            num_edges = data.edge_index.shape[1]
            data.edge_attr = torch.ones((num_edges, 1))  # Example: All edge weights = 1
        data.edge_attr = data.edge_attr.long()
        return data

class EnsureFloatTransform(BaseTransform):
    def __call__(self, data):
        # Ensure node features are float
        if data.x is not None:
            data.x = data.x.float()

        # Ensure edge attributes are float
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr.float()

        # Ensure edge weights are float if they exist
        if hasattr(data, 'edge_weight') and data.edge_weight is not None:
            data.edge_weight = data.edge_weight.float()

        return data


class Train:
    def __init__(self, config, loaders=None):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
        # Dynamically set all variables from config as attributes
        for section, params in config.items():
            if isinstance(params, dict):
                for key, value in params.items():
                    setattr(self, f"{section}_{key}", value)
            else:
                setattr(self, section, params)
        
        # Ensure output directory exists
        os.makedirs(self.out_dir, exist_ok=True)

        if self.dataset_format.startswith('PyG-'):
            self.dataset_class = self.dataset_format.split('-')[1]

            # Special case for GPS model
            if self.model_type == 'GPSModel':
                if self.dataset_name == 'Peptides-func':
                    self.dataset = join_dataset_splits([loader.dataset for loader in loaders])
                elif self.dataset_class == 'TUDataset':
                    self.dataset = ModifiedDataset(join_dataset_splits([loader.dataset for loader in loaders]))
                elif self.dataset_name == 'Cora':
                    
                    self.dataset = join_dataset_splits([loader.dataset for loader in loaders])
                    self.original_dataset = deepcopy(self.dataset).to(self.device)
                else:
                    raise ValueError('not implemented')

            # For MPNN model
            else:
                if self.dataset_class == 'LRGBDataset':
                    # TODO need to check if others also have train/val/test structure
                    if self.dataset_name == 'Peptides-func':
                        self.dataset = join_dataset_splits([LRGBDataset(f'/tmp/{self.dataset_name}', name=self.dataset_name, split=split, transform=EnsureFloatTransform()) for split in ['train', 'val', 'test']])
                    # NOTE This else part has not been tested
                    else:
                        self.dataset = LRGBDataset(root=f'/tmp/{self.dataset_name}', name=self.dataset_name)
                elif self.dataset_class == 'TUDataset':
                    if self.dataset_name == 'IMDB-BINARY':
                        # since IMDB-BINARY does not have node features, node degree is added as node feature
                        self.dataset = CustomDataset(root='/tmp/IMDB-BINARY', name='IMDB-BINARY')
                    else:
                        self.dataset = TUDataset(root=f'/tmp/{self.dataset_name}', name=self.dataset_name)
                elif self.dataset_class == 'Planetoid':
                    self.dataset = Planetoid(root=f'/tmp/{self.dataset_name}', name=self.dataset_name)
                else:
                    raise ValueError(f'{self.dataset_name} from {self.dataset_format} is not supported')
        else:
            raise ValueError('Currently, it only supports dataset from PyG')

        self.nfeat = self.dataset.num_features
        self.nclass = self.dataset.num_classes
        loss_mapping = {
            'cross_entropy': nn.CrossEntropyLoss(),
            'negative_log_likelihood': nn.NLLLoss(),
            'bce': nn.BCEWithLogitsLoss(),
        }
        self.criterion = loss_mapping.get(self.model_loss_fun)
        if self.criterion is None:
            raise ValueError(f'Unsupported loss function: {self.model_loss_fun}')

        model_mapping = {
            'GCN': GCN,
            'GIN': GIN,
            'GAT': GAT,
            'GPSModel': 'GPSModel',
        }
        self.model_class = model_mapping.get(self.model_type)
        if self.model_class is None:
            raise ValueError(f'Unsupported model type: {self.model_type}')


        optimizer_mapping = {
            'adamW': torch.optim.AdamW,
            'adam': torch.optim.Adam
        }
        self.optimizer = optimizer_mapping.get(self.optim_optimizer)
        if self.optimizer is None:
            raise ValueError(f'Unsupported optimizer type: {self.optim_optimizer}')

        metric_mapping = {
            'ap': average_precision_score,
            'f1': f1_score,
            'accuracy': accuracy_score
        }
        self.metric = metric_mapping.get(self.metric_best)
        if self.metric is None:
            raise ValueError(f'Unsupported metric type: {self.metric_best}')

        scheduler_mapping = {
            'cosine_with_warmup': get_cosine_schedule_with_warmup,
            'steplr': torch.optim.lr_scheduler.StepLR,
            'reduce_on_plateau': torch.optim.lr_scheduler.ReduceLROnPlateau
        }
        self.scheduler = scheduler_mapping.get(self.optim_scheduler)
        if self.scheduler is None:
            print(f'Not using a learning rate scheduler. Using {self.optim_base_lr} for the entire training')
            # raise ValueError(f'Unsupported scheduler type: {self.optim_scheduler}')

        # if there is only 1 graph in self.dataset eg. Cora from PyG-Planetoid
        if self.dataset_task == 'node':
            self.dataset = self.dataset[0]
            self.dataset_length = self.dataset.y.size(0)
        else:
            self.dataset_length = len(self.dataset)


    def __call__(self):
        if self.dataset_name in ['Peptides-func', 'ENZYMES', 'IMDB-BINARY', 'Cora']:
            if hasattr(self, 'gnn_num_layers') and isinstance(self.gnn_num_layers, list):
                return self.change_num_layer_training()
            return self.training()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def add_default_edge_attr(self, dataset):
        for data in dataset:
            if data.edge_attr is None:
                num_edges = data.edge_index.shape[1]
                data.edge_attr = torch.ones((num_edges, 1))  # Example: All edge weights = 1
            # Ensure edge_attr has the correct shape for your model
            data.edge_attr = data.edge_attr.float()
        return dataset

    def plot_training_metrics(self, train_history, val_history, val_accs, base_filename):
        # Training and Validation Loss
        plt.figure(figsize=(10, 5))
        plt.plot(train_history, label='Training Loss')
        plt.plot(val_history, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'{base_filename} Training and Validation Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'figures/{base_filename}_loss.png')
        plt.close()

        # Validation Accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(val_accs, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'{base_filename} Validation Accuracy')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f'figures/{base_filename}_accuracy.png')
        plt.close()


    def change_num_layer_training(self):
        all_train_losses, all_val_losses, all_test_losses, all_train_metrics, all_val_metrics, all_test_metrics = {}, {}, {}, {}, {}, {}
        for num_layer in self.gnn_num_layers:
            self.gnn_num_layer = num_layer
            train_losses, val_losses, test_losses, train_metrics, val_metrics, test_metrics = self.training()
            all_train_losses[num_layer] = train_losses
            all_val_losses[num_layer] = val_losses
            all_test_losses[num_layer] = test_losses
            all_train_metrics[num_layer] = train_metrics
            all_val_metrics[num_layer] = val_metrics
            all_test_metrics[num_layer] = test_metrics

        return all_train_losses, all_val_losses, all_test_losses, all_train_metrics, all_val_metrics, all_test_metrics


    def training(self):
        all_train_losses, all_val_losses, all_test_losses = {}, {}, {}
        all_train_metrics, all_val_metrics, all_test_metrics = {}, {}, {}

        for i in range(self.train_repetition):
            print(f'{i}-th iteration on {self.dataset_name} with {self.model_type}')
            # TODO add dropout later
            if self.model_type == 'GPSModel':
                model = create_model()
            else:
                model = self.model_class(
                    self.nfeat, 
                    self.gnn_nhid, 
                    self.nclass, 
                    self.gnn_num_layer, 
                    heads=self.gnn_heads
                ).to(self.device)
            optimizer = self.optimizer(model.parameters(), lr=self.optim_base_lr)
            early_stopping = EarlyStoppingLoss(patience=30, fname=f'{self.dataset_name}_{self.model_type}_{self.gnn_num_layer}_layers_best_model.pth')
            # TODO need a dedicated function to handle different scheduler with different configurations
            if self.optim_scheduler == 'steplr':
                scheduler = self.scheduler(
                    optimizer,
                    step_size=self.optim_step_size,
                    gamma=self.optim_gamma
                )
            elif self.optim_scheduler == 'cosine_with_warmup':
                scheduler = self.scheduler(
                    optimizer,
                    num_warmup_steps=self.optim_warmup_steps, 
                    num_training_steps=self.train_epochs
                )
            elif self.optim_scheduler == 'reduce_on_plateau':
                scheduler = self.scheduler(
                    optimizer,
                    mode='max',
                    factor=self.optim_reduce_factor,
                    patience=self.optim_schedule_patience,
                    min_lr=self.optim_min_lr
                )

            train_size = int(self.train_split_ratios[0] * self.dataset_length)
            val_size = int(self.train_split_ratios[1] * self.dataset_length)
            test_size = self.dataset_length - train_size - val_size

            # For Cora from PyG-Planetoid
            if self.dataset_task == 'node':
                self.dataset.to(self.device)
                indices = torch.randperm(self.dataset_length)
                train_indices = indices[:train_size]
                val_indices = indices[train_size:train_size + val_size]
                test_indices = indices[train_size + val_size:]

                # Initialize the masks
                self.dataset.train_mask = torch.zeros(self.dataset_length, dtype=torch.bool).to(self.device)
                self.dataset.val_mask = torch.zeros(self.dataset_length, dtype=torch.bool).to(self.device)
                self.dataset.test_mask = torch.zeros(self.dataset_length, dtype=torch.bool).to(self.device)

                # Set the masks based on the split indices
                self.dataset.train_mask[train_indices] = True
                self.dataset.val_mask[val_indices] = True
                self.dataset.test_mask[test_indices] = True
            # For Enzymes, IMDB-Binary, Peptides-func
            else:
                train_set, val_set, test_set = random_split(
                     self.dataset, [train_size, val_size, test_size]
                )

    
                train_loader = DataLoader(train_set, batch_size=self.train_batch_size, shuffle=True)
                val_loader = DataLoader(val_set, batch_size=self.train_batch_size, shuffle=False)
                test_loader = DataLoader(test_set, batch_size=self.train_batch_size, shuffle=False)
    
            train_losses, val_losses, test_losses = [], [], []
            train_metrics, val_metrics, test_metrics = [], [], []
        
            for epoch in range(self.train_epochs):
                if self.dataset_name == 'Cora' and self.model_type == 'GPSModel':
                    self.dataset.x = self.original_dataset[0].x.clone()
                model.train()
                total_loss = 0
                y_true_train, y_pred_train = [], []

                if self.dataset_task == 'node':
                    optimizer.zero_grad()
                    if self.model_type == 'GPSModel':
                        out, _ = model(self.dataset)
                    else:
                        out = model(self.dataset)

                    train_loss, train_metric = self.calc_loss_metric(out, self.dataset.train_mask)
                    val_loss, val_metric = self.calc_loss_metric(out, self.dataset.val_mask)
                    test_loss, test_metric = self.calc_loss_metric(out, self.dataset.test_mask)
                    
                    train_losses.append(train_loss.item())
                    train_metrics.append(train_metric)
                    val_losses.append(val_loss.item())
                    val_metrics.append(val_metric)
                    test_losses.append(test_loss.item())
                    test_metrics.append(test_metric)

                    train_loss.backward()
                    optimizer.step()

                elif self.dataset_task == 'graph':
                    # Training loop
                    for batch in train_loader:
                        batch.x = batch.x.float()
                        batch = batch.to(self.device)
        
                        optimizer.zero_grad()
        
                        # Forward pass
                        if self.model_type == 'GPSModel':
                            out = model(batch)
                            out, _ = out
                            if self.model_loss_fun == 'bce':
                                out = out.squeeze()
                                batch.y = batch.y.float()  
                        else:
                            out = model(batch, task=self.dataset_task)
                        # print(batch.y)
                        # print(self.criterion)
                        # print('out: ', out)
                        loss = self.criterion(out, batch.y)
                        # print(loss.shape)
                        loss.backward()
                        optimizer.step()
        
                        total_loss += loss.item()
                        y_true_train.append(batch.y.cpu())
                        y_pred_train.append(out.detach().cpu())
        
                    train_losses.append(total_loss / len(train_loader))
                    train_metrics.append(self.calc_metric(y_true_train, y_pred_train))

                    # Validation loop
                    model.eval()
                    val_loss = 0.0
                    y_true_val, y_pred_val = [], []
                    test_loss = 0.0
                    y_true_test, y_pred_test = [], []
                    with torch.no_grad():
                        for batch in val_loader:
                            batch = batch.to(self.device)
                            if self.model_type == 'GPSModel':
                                out = model(batch)
                                out, _ = out
                                if self.model_loss_fun == 'bce':
                                    out = out.squeeze()
                                    batch.y = batch.y.float()  
                            else:
                                out = model(batch, task=self.dataset_task)
                            loss = self.criterion(out, batch.y)
                            val_loss += loss.item()
                            y_true_val.append(batch.y.cpu())
                            y_pred_val.append(out.detach().cpu())
        
                        for batch in test_loader:
                            batch = batch.to(self.device)
                            if self.model_type == 'GPSModel':
                                out = model(batch)
                                out, _ = out
                                if self.model_loss_fun == 'bce':
                                    out = out.squeeze()
                                    batch.y = batch.y.float()  
                            else:
                                out = model(batch, task=self.dataset_task)
                            loss = self.criterion(out, batch.y)
                            test_loss += loss.item()
                            y_true_test.append(batch.y.cpu())
                            y_pred_test.append(out.detach().cpu())
    
                    val_losses.append(val_loss)
                    test_losses.append(test_loss)
                    val_metric = self.calc_metric(y_true_val, y_pred_val)
                    test_metric = self.calc_metric(y_true_test, y_pred_test)
        
                    val_metrics.append(val_metric)
                    test_metrics.append(test_metric)
                # elif link prediction task?
                else:
                    raise ValueError(f'Unsupported task: {self.dataset_task}')

                if epoch%1 == 0:
                    print('epoch:', epoch+1)
                    print(f'train_{self.metric_best}:', train_metrics[-1])
                    print(f'val_{self.metric_best}:', val_metrics[-1])
                    print(f'test_{self.metric_best}:', test_metrics[-1])

                if self.optim_scheduler == 'reduce_on_plateau':
                    scheduler.step(val_metric)
                elif self.optim_scheduler:
                    scheduler.step()
                # if `self.optim_scheduler` is not set
                
                # Early stopping
                # NOTE changed to val_metric
                early_stopping(val_metric, model)
                if early_stopping.early_stop:
                    print("Early stopping triggered.")
                    break

            all_train_losses[i] = train_losses
            all_val_losses[i] = val_losses
            all_test_losses[i] = test_losses
            all_train_metrics[i] = train_metrics
            all_val_metrics[i] = val_metrics
            all_test_metrics[i] = test_metrics

        return all_train_losses, all_val_losses, all_test_losses, all_train_metrics, all_val_metrics, all_test_metrics


    def calc_metric(self, y_true, y_pred):
        # if using batch loading when doing graph classification
        if self.dataset_task == 'graph':
            y_true = torch.cat(y_true, dim=0)
            y_pred = torch.cat(y_pred, dim=0)

        if self.metric_best == 'ap':
            if self.dataset_task_type == "classification_multilabel":
                # TODO fix this later
                y_pred = torch.sigmoid(y_pred)
        elif self.metric_best == 'accuracy':  # Accuracy
            # Handle binary vs multi-class classification (using bce or ce)
            if y_pred.ndim == 1 or (y_pred.ndim == 2 and y_pred.shape[1] == 1):
                # Binary classification: apply sigmoid and threshold
                y_pred = (torch.sigmoid(y_pred) > 0.5).long()
            else:
                # Multi-class classification: take argmax across classes
                # even though imdb is binary, gcn, gin, gat are using cross entropy loss
                y_pred = torch.argmax(y_pred, dim=1)
        # elif self.metric_best == 'accuracy':
        #     y_pred = torch.argmax(y_pred, dim=1)

        metric = self.metric(y_true.cpu().numpy(), y_pred.cpu().numpy())
        return metric

    
    # def calc_loss_metric(self, out, mask):
    #     mask_out = out[mask]
    #     print(mask_out)
    #     print(self.dataset.y[mask])
    #     mask_loss = self.criterion(
    #         mask_out, self.dataset.y[mask]
    #     )
    #     mask_metric = self.calc_metric(
    #         self.dataset.y[mask], mask_out
    #     )
    #     return mask_loss, mask_metric

    def calc_loss_metric(self, out, mask):
        # Check if the model type requires log_softmax
        if self.model_type == 'GPSModel':
            out = F.log_softmax(out, dim=1)  # Apply log_softmax for NLLLoss
    
        # Mask the output and labels
        mask_out = out[mask]
    
        # Calculate the loss using the criterion
        mask_loss = self.criterion(mask_out, self.dataset.y[mask])
    
        # Calculate the metric (e.g., accuracy)
        mask_metric = self.calc_metric(self.dataset.y[mask], mask_out)
    
        return mask_loss, mask_metric


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


def plot_accuracy(result, dataset_name):
    plt.figure(figsize=(10, 5))
    num_layers = range(1, len(result['GCN']['val_accs']) + 1)

    # Define colors for each model
    model_colors = {
        'GCN': 'blue',
        'GIN': 'red',
        'GAT': 'green'
    }

    for model_name, color in model_colors.items():
        plt.plot(num_layers, result[model_name]['train_accs'], label=f'{model_name} Train Acc', linestyle=':', color=color, alpha=0.5)
        plt.plot(num_layers, result[model_name]['val_accs'], label=f'{model_name} Val Acc', linestyle='--', color=color)
        plt.plot(num_layers, result[model_name]['test_accs'], label=f'{model_name} Test Acc', linestyle='-', color=color)

    plt.xlabel('Number of Layers')
    plt.ylabel('Accuracy')
    plt.xticks(num_layers)
    plt.title(f'Accuracies for GCN, GIN, GAT on {dataset_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig(f'figures/{dataset_name}_accuracy_histories.png', format='png', dpi=300, bbox_inches="tight")
    plt.close()


def changing_num_layers(dataset, task, model_class, nhid=64, heads=4, lr=0.001, batch_size=64, epochs=500):
    trainer = Train(dataset=dataset, task=task, verbose=False)
    train_accs, val_accs, test_accs = [], [], []
    for num_layer in range(1, 16):
        print(num_layer)
        train_tmp, val_tmp, test_tmp = [], [], []
        for _ in range(30):
            _, _, _, train_acc, val_acc, test_acc = trainer(
                model_class=model_class,
                nhid=nhid,
                heads=heads,
                mlp_num=num_layer,
                lr=lr,
                batch_size=batch_size,
                epochs=epochs
            )
            train_tmp.append(train_acc)
            val_tmp.append(val_acc)
            test_tmp.append(test_acc)

        train_accs.append(np.median(train_tmp))
        val_accs.append(np.median(val_tmp))
        test_accs.append(np.median(test_tmp))

    return train_accs, val_accs, test_accs


def plot_boxplot(result):
    datasets = list(result.keys())
    models = list(result[datasets[0]].keys())
    metrics = list(result[datasets[0]][models[0]].keys())
    
    # Colors for train, validation, and test
    metric_colors = {
        'train_accs': 'skyblue',
        'val_accs': 'salmon',
        'test_accs': 'lightgreen'
    }

    # Create directory for CSVs if it doesn't exist
    os.makedirs("results_csv", exist_ok=True)

    fig, axes = plt.subplots(1, len(datasets), figsize=(18, 6), sharey=True)
    fig.suptitle('Performance Distribution Across Datasets', fontsize=16)

    csv_data = []  # Initialize data storage for CSV

    for i, (dataset, ax) in enumerate(zip(datasets, axes)):
        data_to_plot = []
        labels = []
        positions = []
        pos = 1

        for model in models:
            model_data = []  # Holds data for the current model
            for metric in metrics:
                model_data.append(result[dataset][model][metric])
                csv_data.append({
                    'Dataset': dataset,
                    'Model': model,
                    'Metric': metric,
                    'Values': result[dataset][model][metric]
                })

            # Append model data to the main list and add positions
            data_to_plot.extend(model_data)
            labels.append(model)  # Label model once
            positions.extend([pos, pos + 1, pos + 2])  # Position each metric under its model group
            pos += 4  # Leave space for separation between model groups

        # Plot each metric with different colors
        bp = ax.boxplot(data_to_plot, patch_artist=True, positions=positions)
        for patch, metric in zip(bp['boxes'], metrics * len(models)):
            patch.set_facecolor(metric_colors[metric])

        # Draw dividing lines between models
        for j in range(1, len(models)):
            ax.axvline(x=j * 4 - 0.5, color='grey', linestyle='--')

        # Set x-tick labels to model names only, centered under each set of boxplots
        ax.set_xticks([j * 4 + 1 for j in range(len(models))])
        ax.set_xticklabels(models, rotation=45, ha='right')

        # Axis labels and limits
        ax.set_title(f"{dataset}", fontsize=14)
        ax.set_xlabel("Model")
        ax.set_ylim(0, 1)
    
    # Set y-axis label for the entire figure
    axes[0].set_ylabel('Accuracy')
    
    # Save each dataset's CSV data
    for dataset in datasets:
        df = pd.DataFrame([entry for entry in csv_data if entry['Dataset'] == dataset])
        df.to_csv(f'results_csv/{dataset}_accuracy_results.csv', index=False)

    # Save and display
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'figures/combined_boxplot.png', format='png', dpi=300)
    plt.close()


def join_dataset_splits(datasets):
    """Join train, val, test datasets into one dataset object.

    Args:
        datasets: list of 3 PyG datasets to merge

    Returns:
        joint dataset with `split_idxs` property storing the split indices
    """
    assert len(datasets) == 3, "Expecting train, val, test datasets"

    n1, n2, n3 = len(datasets[0]), len(datasets[1]), len(datasets[2])
    data_list = []

    for i, dataset in enumerate(datasets):
        split_name = ['train', 'val', 'test'][i]  # Identify the current split
        for j in range(len(dataset)):
            data = dataset.get(j)
            
            # Ensure 'y' attribute exists and is not None
            if not hasattr(data, 'y') or data.y is None:
                raise ValueError(f"Missing 'y' attribute in {split_name} dataset at index {j}.")
            
            data_list.append(data)

    datasets[0]._indices = None
    datasets[0]._data_list = data_list
    datasets[0].data, datasets[0].slices = datasets[0].collate(data_list)

    split_idxs = [
        list(range(n1)),
        list(range(n1, n1 + n2)),
        list(range(n1 + n2, n1 + n2 + n3))
    ]
    datasets[0].split_idxs = split_idxs

    return datasets[0]
