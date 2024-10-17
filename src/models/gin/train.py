from sklearn.model_selection import KFold
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
import numpy as np
from gin import GIN


# TODO use EarlyStopping?
def train_model(dataset, criterion, nfeat, nclass, task, device, lr=0.001, k=10, nhid=64, mlp_num=2, epoch=300):
    """
    Args:
        nfeat (int): number of node features
        nclass (int): number of class labels
        task (str): `node` or `graph` for node/graph classification
        lr (float): learning rate for optimizer
        k (int): number of folds for K-Fold
        nhid (int): size of hidden dimension in GIN's MLP
        mlp_num (int): number of MLP layers to include
        epoch (int): number of iterations to train a model
    """
    kfold = KFold(n_splits=k, shuffle=True)
    accuracies = []

    for fold, (train_index, test_index) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold + 1}')
        model = GIN(nfeat, nhid, nclass, mlp_num).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_dataset = Subset(dataset, train_index)
        test_dataset = Subset(dataset, test_index)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        for epoch in range(epoch):
            model.train()
            for data in train_loader:
                data = data.to(device)
                out = model(data, task=task)
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        model.eval()
        correct = 0
        for data in test_loader:
            data = data.to(device)
            out = model(data, task=task)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
        test_acc = correct / len(test_loader.dataset)
        print(f'Test Accuracy: {test_acc}')
        accuracies.append(test_acc)

    print(f'\nAverage Test Accuracy over 10 folds: {np.mean(accuracies)}')

    return accuracies
