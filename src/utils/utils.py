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
from sklearn.metrics import f1_score, average_precision_score
from torch_geometric.transforms import BaseTransform
from ..models.gcn import GCN
from ..models.gin import GIN
from ..models.gat import GAT


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
    def __init__(self, config):
        self.config = config
    
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
            if self.dataset_class == 'LRGBDataset':
                # TODO need to check if others also have train/val/test structure
                if self.dataset_name == 'Peptides-func':
                    self.dataset = join_dataset_splits([LRGBDataset(f'/tmp/{self.dataset_name}', name=self.dataset_name, split=split, transform=EnsureFloatTransform()) for split in ['train', 'val', 'test']])
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

        if self.model_loss_fun == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError('Please specify pre-defined loss function. Available loss function(s) are cross_entropy')

        self.nfeat = self.dataset.num_features
        self.nclass = self.dataset.num_classes
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model_mapping = {
            'GCN': GCN,
            'GIN': GIN,
            'GAT': GAT,
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


    def __call__(self):
        model = self.model_class(self.nfeat, self.gnn_nhid, self.nclass, self.gnn_num_layer, heads=self.gnn_heads).to(self.device)
        optimizer = self.optimizer(model.parameters(), lr=self.optim_base_lr)
        early_stopping = EarlyStoppingLoss(patience=10, fname=f'{self.dataset_name}_{self.model_type}_{self.gnn_num_layer}_layers_best_model.pth')
        max_retries = 5
        retry_delay = 0.1

        if self.dataset_name == 'CORA':
            return self.cora_training(model, optimizer, early_stopping, epochs)
        elif self.dataset_name == 'Peptides-func':
            return self.peptides_func_training(model, optimizer, early_stopping)

        train_history, val_history, val_accs = [], [], []

        # 7:1.5:1.5 split
        train_size = int(0.7 * len(self.dataset))
        val_size = int(0.15 * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size],
#            generator=torch.Generator().manual_seed(42)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Training with early stopping
        # TODO clean up training epochs
        for epoch in range(epochs):
            # Training phase
            model.train()
            total_loss = 0
            for data in train_loader:
                data = data.to(self.device)
                out = model(data, task=self.task)
                loss = self.criterion(out, data.y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            # Validation phase
            model.eval()
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(self.device)
                    out = model(data, task=self.task)
                    loss = self.criterion(out, data.y)
                    val_loss += loss.item()
                    pred = out.argmax(dim=1)
                    correct += int((pred == data.y).sum())

            val_loss /= len(val_loader)
            val_acc = correct / len(val_loader.dataset)
            val_accs.append(val_acc)
            train_history.append(total_loss / len(train_loader))
            val_history.append(val_loss)

            if self.verbose and epoch%20==0:
                print(f'Epoch {epoch + 1}, Training Loss: {total_loss / len(train_loader)}, Validation Loss: {val_loss}, Validation Accuracy: {val_acc}')

            # Early stopping
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                # print(f'Early stopping triggered after {epoch + 1} epochs.')
                break

        # Load the best model and evaluate on the test set
        #model.load_state_dict(torch.load(f'{early_stopping.prefix_path}/{early_stopping.fname}', weights_only=True))
        best_model_state = early_stopping.get_best_model()
        model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            correct = 0
            for data in test_loader:
                data = data.to(self.device)
                out = model(data, task=self.task)
                pred = out.argmax(dim=1)
                correct += int((pred == data.y).sum())
            test_acc = correct / len(test_loader.dataset)

            correct = 0
            for data in train_loader:
                data = data.to(self.device)
                out = model(data, task=self.task)
                pred = out.argmax(dim=1)
                correct += int((pred == data.y).sum())
            train_acc = correct / len(train_loader.dataset)

            correct = 0
            for data in val_loader:
                data = data.to(self.device)
                out = model(data, task=self.task)
                pred = out.argmax(dim=1)
                correct += int((pred == data.y).sum())
            val_acc = correct / len(val_loader.dataset)

        if self.plot:
            self.plot_training_metrics(train_history, val_history, val_accs, f'{self.dataset_name}_{model._get_name()}')
        return train_history, val_history, val_accs, train_acc, val_acc, test_acc

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

    def peptides_func_training(self, model, optimizer, early_stopping):
        train_indices = self.dataset.split_idxs[0]
        val_indices = self.dataset.split_idxs[1]
        test_indices = self.dataset.split_idxs[2]

        train_loader = DataLoader([self.dataset.get(i) for i in train_indices], batch_size=self.train_batch_size, shuffle=True)
        val_loader = DataLoader([self.dataset.get(i) for i in val_indices], batch_size=self.train_batch_size, shuffle=False)
        test_loader = DataLoader([self.dataset.get(i) for i in test_indices], batch_size=self.train_batch_size, shuffle=False)

    
        train_losses, val_losses, test_losses = [], [], []
        train_aps, val_aps, test_aps = [], [], []
    
        for epoch in range(self.train_epochs):
            model.train()
            total_loss = 0
            y_true_train, y_pred_train = [], []

            # Training loop
            for batch in train_loader:
                batch.x = batch.x.float()
                batch = batch.to(self.device)  # Move batch to device

                optimizer.zero_grad()

                # Forward pass
                out = model(batch, task=self.dataset_task)
                loss = self.criterion(out, batch.y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                y_true_train.append(batch.y.cpu())
                y_pred_train.append(out.detach().cpu())

            train_losses.append(total_loss / len(train_loader))

            y_true_train = torch.cat(y_true_train, dim=0)
            y_pred_train = torch.cat(y_pred_train, dim=0)
            train_ap = average_precision_score(y_true_train.numpy(), y_pred_train.numpy(), average='macro')
            train_aps.append(train_ap)

            model.eval()
            val_loss = 0
            y_true_val, y_pred_val = [], []
            test_loss = 0
            y_true_test, y_pred_test = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    out = model(batch, task=self.dataset_task)
                    loss = self.criterion(out, batch.y)
                    val_loss += loss.item()
                    y_true_val.append(batch.y.cpu())
                    y_pred_val.append(out.detach().cpu())

                for batch in test_loader:
                    batch = batch.to(self.device)
                    out = model(batch, task=self.dataset_task)
                    loss = self.criterion(out, batch.y)
                    test_loss += loss.item()
                    y_true_test.append(batch.y.cpu())
                    y_pred_test.append(out.detach().cpu())

            val_losses.append(val_loss / len(val_loader))
            test_losses.append(test_loss / len(test_loader))

            y_true_val = torch.cat(y_true_val, dim=0)
            y_pred_val = torch.cat(y_pred_val, dim=0)
            val_ap = average_precision_score(y_true_val, y_pred_val, average='macro')
            val_aps.append(val_ap)

            y_true_test = torch.cat(y_true_test, dim=0)
            y_pred_test = torch.cat(y_pred_test, dim=0)
            test_ap = average_precision_score(y_true_test, y_pred_test, average='macro')
            test_aps.append(test_ap)

            print('epoch:', epoch)
            print('train_ap:', train_ap)
            print('val_ap:', val_ap)
            print('test_ap:', test_ap)

            # Early stopping
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break


        return train_losses, val_losses, test_losses, train_aps, val_aps, test_aps



    def cora_training(self, model, optimizer, early_stopping, epochs):
        data = self.dataset[0].to(self.device)
        train_history, val_history, val_accs = [], [], []
        
        # Set the split ratios for training, validation, and test sets
        num_nodes = data.y.size(0)
        train_size = int(0.7 * num_nodes)
        val_size = int(0.15 * num_nodes)
        test_size = num_nodes - train_size - val_size
        
        # Randomly split indices for each mask
        indices = torch.randperm(num_nodes)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Initialize the masks
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool).to(self.device)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool).to(self.device)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool).to(self.device)
        
        # Set the masks based on the split indices
        data.train_mask[train_indices] = True
        data.val_mask[val_indices] = True
        data.test_mask[test_indices] = True
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            pred = out[data.val_mask].argmax(dim=1)
            correct = int((pred == data.y[data.val_mask]).sum())
            train_history.append(loss.item())
            val_accs.append(correct / len(pred))
            loss.backward()
            optimizer.step()

            if self.verbose and epoch % 10 == 0:
                print(f'{epoch}: {loss.item()}')

            model.eval()
            with torch.no_grad():
                val_loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
                val_history.append(val_loss.item())
                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                    # print(f'Early stopping triggered after {epoch+1} epochs.')
                    break

        #model.load_state_dict(torch.load(f'{early_stopping.prefix_path}/{early_stopping.fname}', weights_only=True))
        best_model_state = early_stopping.get_best_model()
        model.load_state_dict(best_model_state)
        model.eval()
        pred = model(data).argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        test_acc = int(correct) / len(pred[data.test_mask])

        correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
        val_acc = int(correct) / len(pred[data.val_mask])

        correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
        train_acc = int(correct) / len(pred[data.train_mask])
        # print(f'Test Accuracy: {test_acc}')
        if self.plot:
            self.plot_training_metrics(train_history, val_history, val_accs, f'{self.dataset_name}_{model._get_name()}')
        return train_history, val_history, val_accs, train_acc, val_acc, test_acc


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


def default_test(dataset, task, model_class, num_iterations=30, num_layer=2, nhid=64, heads=4, lr=0.001, batch_size=64, epochs=500):
    trainer = Train(dataset=dataset, task=task, verbose=False)
    train_accs, val_accs, test_accs = [], [], []
    for _ in range(num_iterations):
        _, _, _, train_acc, val_acc, test_acc = trainer(
            model_class=model_class,
            nhid=nhid,
            heads=heads,
            mlp_num=num_layer,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs
        )
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        train_accs.append(train_acc)

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


def convert_edge_weights(data):
    if data.edge_weight is not None:
        data.edge_weight = data.edge_weight.float()  # Convert edge weights to float
    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr.float()  # Convert edge attributes to float
    return data

def normalize_edge_weights(data):
    # Example: Normalize edge weights or ensure degrees are float
    row, col = data.edge_index
    deg = degree(row, data.num_nodes, dtype=torch.float)  # Ensure degree is float
    return data

def compute_f1(pred, true):
    pred = pred.cpu().numpy()
    true = true.cpu().numpy()
    return f1_score(true, pred, average='macro')