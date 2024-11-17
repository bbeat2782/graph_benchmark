import numpy as np
import torch


class EarlyStoppingLoss:
    """
    Early stops the training if validation loss doesn't
    improve after a given patience.
    """

    def __init__(self, fname='es_checkpoint.pth', patience=5, verbose=False, delta=0, prefix='model_chkt_pnt'):
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
        self.val_loss_min = np.inf
        self.delta = delta
        self.prefix_path = prefix
        self.fname = fname
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = val_loss

        if self.best_score is None or score < self.best_score - self.delta:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0
            if self.verbose:
                print(f"Validation loss improved to {val_loss:.6f}. Saving best model state.")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        
    def get_best_model(self):
        return self.best_model_state
