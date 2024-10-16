import numpy as np
import torch


class EarlyStoppingLoss:
    """
    Early stops the training if validation loss doesn't
    improve after a given patience.
    """

    def __init__(self, fname='es_checkpoint.pth', patience=5, verbose=False, delta=0, prefix=None):
        """
        Args:
            fname (str)   : Checkpoint file name
            patience (int): How long to wait after last time validation loss improved
            verbose (bool): If True, prints a message for each validation loss improvement
            delta (float) : Minimum change in the monitored quantity to qualify as an improvement
            prefix (str)  : Path to store the best model
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.prefix_path = prefix
        self.fname = fname

    def __call__(self, val_loss, model):
        score = val_loss

        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping Loss counter: {self.counter} out of {self.patience}')
            print(f'Now loss:{val_loss}\tBest_loss:{self.best_score}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.prefix_path + self.fname)
        self.val_loss_min = val_loss
