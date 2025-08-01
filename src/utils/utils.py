import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import logging

from collections import deque
from dataclasses import dataclass, field
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import (
    BinaryAccuracy, BinaryAUROC, BinaryF1Score, BinaryMatthewsCorrCoef,
    MulticlassAccuracy, MulticlassAUROC, MulticlassF1Score)
from torchmetrics.functional import f1_score
from torchmetrics import MetricCollection
from typing import Optional

logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # Convert tagers to float if needed
        targets = targets.float()
        
        # Compute the probability
        probs = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_term = (1 - pt) ** self.gamma
        
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_t * focal_term * bce_loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, metric):
        if self.best_score is None or metric > self.best_score + self.min_delta:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class RollingEarlyStopping(EarlyStopping):
    def __init__(self, patience=10, window=5, min_delta=1e-3):
        super().__init__(
            patience=patience,
            min_delta=min_delta,
        )
        self.history=deque(maxlen=window)
        self.window=window
        
        self.best_avg = None
        self.prev_model_state = None
        self.last_model_state = None
        
    def __call__(self, metric, model):
        self.history.append(metric)
        self.prev_model_state = self.last_model_state
        self.last_model_state = copy.deepcopy(model.state_dict())

        if len(self.history) < self.history.maxlen:
            return # Window must be full
        
        current_avg = sum(self.history) / len(self.history)
        
        if self.best_avg is None or current_avg > self.best_avg + self.min_delta:
            self.best_avg = current_avg
            return
        
        if current_avg > self.best_avg + self.min_delta:
            self.best_avg = current_avg
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    
    def restore_prior_model(self, model):
        if self.prev_model_state is not None:
            model.load_state_dict(self.prev_model_state)
    
def collate_with_macro(batch):
    """
    Custom collate_fn that stacks firm_seq and label normally,
    and replicates macro_seq across the batch.
    """
    firm_seq = torch.stack([item["firm_seq"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    
    # We assume all macro_seq are identical (shared), so take the first
    macro_seq = batch[0]["macro_seq"].T
    
    # Expand macro_seq to match batch size
    macro_seq = macro_seq.unsqueeze(0).expand(len(batch), -1, -1)

    return {
        "firm_seq": firm_seq,
        "macro_seq": macro_seq,
        "label": labels
    }

def find_best_threshold(model, val_loader, device, n_steps=101):
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            firm_x = batch["firm_seq"].to(device)
            macro_x = batch["macro_seq"].to(device)
            labels = batch["label"].to(device).squeeze(-1).int()

            logits = model(firm_x, macro_x)
            if isinstance(logits, tuple):
                logits = logits[0] # Handle TFT case
            
            probs = torch.sigmoid(logits).squeeze(-1)
            
            all_probs.append(probs)
            all_labels.append(labels.squeeze())

    probs = torch.cat(all_probs, dim=0)
    labels = torch.cat(all_labels, dim=0)

    best_f1 = -1.0
    best_threshold=0.5
    
    thresholds = torch.linspace(0, 1, steps=n_steps).to(device)
    for t in thresholds:
        preds = (probs >= t).int().squeeze(-1)
        f1 = f1_score(preds, labels, task="binary")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t.item()

    return best_threshold, best_f1

class CustomReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self, 
                optimizer: torch.optim.Optimizer,
                mode: str="min", 
                factor: float=0.25, 
                patience: int=5,
                threshold: float=1e-4, 
                threshold_mode: str='rel',
                cooldown: int=10, 
                min_lr: float=1e-6, 
                eps: float=1e-8,
                rate_threshold: float=1e-4, 
                rate_patience: int=2):
        """
        Extended ReduceLROnPlateau scheduler that reduces LR on plateau _and_ when the 
        rate of decrease in training loss slows down.

        Args:
            optimizer (Optimizer): Wrapped optimizer. Variables mode, factor, patience, 
                                    threshold, cooldown, min_lr and eps are inherited 
                                    from ReduceLROnPlateau.
            rate_threshold (float): Minimum rate of decrease in loss to avoid triggering
                                    an LR reduction. Default: 1e-4.
            rate_patience (int): Number of epochs to calculate the average rate of 
                                    decrease. Default: 2.   
        """
        super().__init__(optimizer, mode, factor, patience, threshold, 
                        threshold_mode, cooldown, min_lr, eps)
        self.rate_threshold = rate_threshold
        self.rate_patience = rate_patience
        self.loss_history = []

    def step(self, metric: float, epoch: Optional[int]=None):
        """
        Monitors the metrics; reduces learning rate based on plateau or slowing rate of decrease.

        Args:
            metric (float): value of the metric to monitor (e.g. train or val loss).
            epoch (int, optional): current epoch. Default: None.
        """
        
        self.loss_history.append(metric) # Update the loss history
        
        if len(self.loss_history) > self.rate_patience:
            delta = self.loss_history[-self.rate_patience] - self.loss_history[-1]
            rate_of_decrease = delta / self.rate_patience
            
            # Cleaning history
            if len(self.loss_history) > self.rate_patience:
                self.loss_history = self.loss_history[-self.rate_patience:]
            
            # Ensure that LR is not reduced during cooldowns
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                return
            
            if rate_of_decrease < self.rate_threshold:
                if self.verbose:
                    print(f"Loss improvement rate {rate_of_decrease:.6f} below threshold "
                        f"{self.rate_threshold}: reducing LR.")
                self._reduce_lr(epoch) # Trigger LR reduction
                self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0
                return
        
        super().step(metric, epoch) # Standard behavior

@dataclass
class TrainConfig:
    firm_data: str
    macro_data: list[str]
    bankruptcy_col: str
    company_col: str
    revenue_cap: int = 3_000
    num_classes: int = 2
    batch_size: int = 32
    epochs: int = 40
    lr: float = 1e-3
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = .2
    threshold: float = 0.5
    alpha: float = 0.9
    gamma: float = 2.0
    scheduler_factor: float = 0.85
    scheduler_patience: int = 50
    stopping_patience: int = 10
    min_lr:float = 0.0
    decay_ih: float = 1e-5
    decay_hh: float = 1e-5
    decay_other: float = 1e-5
    train_fract: float = .8
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    metrics: list[str] = field(default_factory=lambda: ["f1", "accuracy"])
    
    def get_metrics(self) -> MetricCollection:
        """Constructs a MetricCollection from the specified config"""
        if self.num_classes == 2:
            available = {
                "f1": BinaryF1Score(self.threshold),
                "accuracy": BinaryAccuracy(self.threshold),
                "auroc": BinaryAUROC(),
                "matthews": BinaryMatthewsCorrCoef(self.threshold)
            }
        else:
            available = {
                "f1": MulticlassF1Score(num_classes=self.num_classes),
                "accuracy": MulticlassAccuracy(num_classes=self.num_classes),
                "auroc": MulticlassAUROC(num_classes=self.num_classes)
            }
        selected = {k: available[k] for k in self.metrics if k in available}
        return MetricCollection(selected)
    
def normalize_logits(x: torch.Tensor):
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True) + 1e-6
    return (x - mean) / std