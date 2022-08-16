"""
__author__ = 'linlei'
__project__:utils
__time__:2022/6/26 16:43
__email__:"919711601@qq.com"
"""
import torch
import numpy as np


def save_checkpoint(state,path):
    """
    save checkpoint during training
    Args:
        state: consist of model.state_dict(),epoch,loss
        path: save path

    Returns:

    """
    print("saving checkpoint")
    torch.save(state,path)

def load_checkpoint(path):
    """
    load checkpoint
    Args:
        path: checkpoint path

    Returns:
        model_dict,epoch,loss
    """
    checkpoint=torch.load(path)
    model_dict=checkpoint["model_state_dict"]
    epoch=checkpoint["epoch"]
    loss=checkpoint["loss"]
    return  model_dict,epoch,loss

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, state, model):

        score = -state["loss"]

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(state, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(state, model)
            self.counter = 0

    def save_checkpoint(self, state, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        save_checkpoint(state, self.path)
        self.val_loss_min = state["loss"]

def cal_RMSE(pred,real):
    MSE=np.mean((pred-real)**2)
    RMSE=np.sqrt(MSE)
    return RMSE

def cal_R2(pred,real):
    SSR=sum((real-pred)**2)
    SST=sum((real-np.mean(real))**2)
    r2=1-SSR/SST
    return r2