import os
import torch
import numpy as np
from torch import nn
import random
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import torch.nn.functional as F
from .earlystopping import EarlyStoppingLoss
import pandas as pd


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


class Train:
    def __init__(self, dataset, task, criterion='CrossEntropyLoss', verbose=True, plot=False):
        if isinstance(dataset, str):
            self.dataset_name = dataset.upper()
            if self.dataset_name == 'ENZYMES':
                self.dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
            elif self.dataset_name == 'IMDB-BINARY':
                self.dataset = CustomDataset(root='/tmp/IMDB-BINARY', name='IMDB-BINARY')
            elif self.dataset_name == 'CORA':
                self.dataset = Planetoid(root='/tmp/Cora', name='Cora')
            else:
                raise ValueError(f"Dataset {dataset} not supported. Please choose from 'ENZYMES', 'IMDB-BINARY', or 'CORA'.")
        elif isinstance(dataset, CustomDataset):
            self.dataset_name = 'CustomDataset'
            self.dataset = dataset
        else:
            raise ValueError('Please provide appropriate dataset')

        criterion_dict = {
            'CrossEntropyLoss': nn.CrossEntropyLoss(),
            # Other loss functions if I need it
        }

        self.task = task
        self.nfeat = self.dataset.num_features
        self.nclass = self.dataset.num_classes
        if criterion in criterion_dict:
            self.criterion = criterion_dict[criterion]
        else:
            raise ValueError(f"Criterion '{criterion}' not found. Available options are: {list(criterion_dict.keys())}")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using {self.device}')
        self.verbose = verbose
        self.plot = plot

    def __call__(self, model_class, nhid=64, heads=None, mlp_num=2, lr=0.001, batch_size=64, epochs=1000):
        model = model_class(self.nfeat, nhid, self.nclass, mlp_num, heads=heads).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        early_stopping = EarlyStoppingLoss(patience=10, fname=f'{self.dataset_name}_{model._get_name()}_best_model.pth')

        if self.dataset_name == 'CORA':
            return self.cora_training(model, optimizer, early_stopping, epochs)

        train_history, val_history, val_accs = [], [], []

        # 7:1.5:1.5 split
        train_size = int(0.7 * len(self.dataset))
        val_size = int(0.15 * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
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
        model.load_state_dict(torch.load(f'{early_stopping.prefix_path}/{early_stopping.fname}', weights_only=True))
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

    def cora_training(self, model, optimizer, early_stopping, epochs):
        data = self.dataset[0].to(self.device)
        train_history, val_history, val_accs = [], [], []
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

        model.load_state_dict(torch.load(f'{early_stopping.prefix_path}/{early_stopping.fname}', weights_only=True))
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
    plt.title(f'Validation and Test Accuracy for GCN, GIN, GAT on {dataset_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig(f'figures/{dataset_name}_accuracy_histories.png', format='png', dpi=300, bbox_inches="tight")
    plt.close()


def changing_num_layers(dataset, task, model_class, nhid=64, heads=4, lr=0.001, batch_size=64, epochs=500):
    trainer = Train(dataset=dataset, task=task, verbose=False)
    train_accs, val_accs, test_accs = [], [], []
    for num_layer in range(1, 13):
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
