import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    """
    Adjust learning rate based on scheduling strategy.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler: Optional scheduler (if None, uses manual adjustment)
        epoch: Current epoch
        args: Configuration arguments
        printout: Whether to print LR update
    """
    if scheduler is not None:
        # Use built-in scheduler
        scheduler.step()
        if printout:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Learning rate adjusted to {current_lr:.2e}')
        return
    
    # Manual LR adjustment (if no scheduler)
    if args.lradj == 'type1':
        # Exponential decay every epoch
        lr = args.learning_rate * (0.5 ** ((epoch - 1) // 1))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            print(f'Updating learning rate to {lr:.2e}')
    
    elif args.lradj == 'type2':
        # Step decay at specific epochs
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
        if epoch in lr_adjust:
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            if printout:
                print(f'Updating learning rate to {lr:.2e}')


class EarlyStopping:
    """
    Early stopping to halt training when validation loss stops improving.
    """
    def __init__(self, accelerator, patience=7, verbose=False, delta=0):
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'Early stopping triggered after {self.counter} epochs')
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """Save model checkpoint."""
        if self.accelerator.is_local_main_process:
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(model.state_dict(), os.path.join(path, 'checkpoint.pth'))
            if self.verbose:
                print(f'Checkpoint saved at epoch with val_loss: {val_loss:.6f}')
        self.val_loss_min = val_loss


def vali_forecasting(model, vali_loader, criterion, accelerator):
    """
    Validation for forecasting tasks.
    
    Args:
        model: PyTorch model
        vali_loader: Validation data loader
        criterion: Loss function
        accelerator: HuggingFace accelerator
    
    Returns:
        Average validation loss
    """
    total_loss = []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in vali_loader:
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss.append(loss.item())
    
    model.train()
    return np.average(total_loss)


def vali_imputation(model, vali_loader, criterion, accelerator):
    """
    Validation for imputation tasks.
    
    Args:
        model: PyTorch model
        vali_loader: Validation data loader (returns masked, original, mask)
        criterion: Loss function
        accelerator: HuggingFace accelerator
    
    Returns:
        Average validation loss (only on masked positions)
    """
    total_loss = []
    model.eval()
    with torch.no_grad():
        for batch_masked, batch_original, mask in vali_loader:
            batch_masked = batch_masked.float().to(accelerator.device)
            batch_original = batch_original.float().to(accelerator.device)
            mask = mask.float().to(accelerator.device)
            
            outputs = model(batch_masked)
            # Compute loss only on masked positions
            loss = criterion(outputs * mask, batch_original * mask)
            total_loss.append(loss.item())
    
    model.train()
    return np.average(total_loss)


def vali_classification(model, vali_loader, criterion, accelerator):
    """
    Validation for classification tasks.
    
    Args:
        model: PyTorch model
        vali_loader: Validation data loader (returns sequence, label)
        criterion: Loss function
        accelerator: HuggingFace accelerator
    
    Returns:
        Average validation loss
    """
    total_loss = []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in vali_loader:
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.long().to(accelerator.device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss.append(loss.item())
    
    model.train()
    return np.average(total_loss)


def vali_cross_domain(model, vali_loader, criterion, accelerator):
    """
    Validation for cross-domain tasks.
    
    Args:
        model: PyTorch model
        vali_loader: Validation data loader (returns sequence, label)
        criterion: Loss function
        accelerator: HuggingFace accelerator
    
    Returns:
        Average validation loss
    """
    return vali_classification(model, vali_loader, criterion, accelerator)


def metric(pred, true):
    """
    Calculate MSE and MAE metrics.
    
    Args:
        pred: Predictions (numpy array)
        true: Ground truth (numpy array)
    
    Returns:
        mse, mae
    """
    mse = np.mean((pred - true) ** 2)
    mae = np.mean(np.abs(pred - true))
    return mse, mae


def visualize_predictions(pred, true, title="Predictions vs Ground Truth"):
    """
    Simple visualization for predictions.
    
    Args:
        pred: Predictions (numpy array)
        true: Ground truth (numpy array)
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    plt.plot(pred, label='Predictions', color='blue', alpha=0.7)
    plt.plot(true, label='Ground Truth', color='orange', alpha=0.7)
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
