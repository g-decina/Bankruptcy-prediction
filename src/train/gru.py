import mlflow
import logging
import torch
from tqdm import trange

from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryMatthewsCorrCoef

from torch.nn.utils import clip_grad_norm_
from src.utils.utils import RollingEarlyStopping, find_best_threshold

logger = logging.getLogger(__name__)

def train_one_epoch(model, loader, optimizer, loss_fn, device, metrics):
    model.train()
    for metric in metrics.values():
        metric.reset()
    
    total_loss = 0
    for batch in loader:
        firm_seq = batch["firm_seq"].to(device) # (batch, T, F_firm)
        macro_past = batch["macro_past"].to(device) # (T, F_macro) - shared across batch
        labels = batch["label"].to(device) # (batch, 1)

        optimizer.zero_grad()
        preds = model(firm_seq, macro_past) # (batch, 1)
        
        loss = loss_fn(preds, labels)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        for metric in metrics.values():
            for metric in metrics.values():
                if isinstance(metric, (BinaryF1Score, BinaryMatthewsCorrCoef, BinaryAccuracy)):
                    preds_binary = (preds.sigmoid() > 0.5).int()
                    metric.update(preds_binary, labels.int())
                else:
                    metric.update(preds, labels.int())
    
    avg_loss = total_loss / len(loader)
    computed_metrics = {
        name: metric.compute().item() for name, metric in metrics.items()
    }
    computed_metrics["loss"] = avg_loss
    
    return avg_loss, computed_metrics

def evaluate_one_epoch(model, loader, loss_fn, device, metrics):
    model.eval()
    for metric in metrics.values():
        metric.reset()
    
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            firm_seq = batch["firm_seq"].to(device)
            macro_past = batch["macro_past"].to(device)
            labels = batch["label"].to(device)
            
            preds = model(firm_seq, macro_past)
            
            loss = loss_fn(preds, labels)
            total_loss += loss.item()
            
            for metric in metrics.values():
                if isinstance(metric, (BinaryF1Score, BinaryMatthewsCorrCoef, BinaryAccuracy)):
                    preds_binary = (preds.sigmoid() > 0.5).int()
                    metric.update(preds_binary, labels.int())
                else:
                    metric.update(preds, labels.int())
                
    avg_loss = total_loss / len(loader)
    computed_metrics = {
        name: metric.compute().item() for name, metric in metrics.items()
    }
    computed_metrics["loss"] = avg_loss
    
    return computed_metrics
    
def train_gru(
    model, train_loader, val_loader, loss_fn, optimizer,
    scheduler, stopping_patience, stopping_window, device, epochs, 
    metrics
):
    progress_bar = trange(epochs, desc = "Training", leave = True)
    early_stopping = RollingEarlyStopping(
        patience=stopping_patience, window=stopping_window
    )
    
    for epoch in progress_bar:
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, metrics
        )
        
        last_lr = scheduler.get_last_lr()[0]
        train_metrics["lr"] = last_lr
        
        for name, value in train_metrics.items():
            mlflow.log_metric(f"train_{name}", value, step = epoch)
            
        val_metrics = evaluate_one_epoch(model, val_loader, loss_fn, device, metrics)
        if "matthews" in val_metrics:
            early_stopping(metric=val_metrics["matthews"], model=model)
            
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}, restoring model from epoch {epoch - 1}")
            early_stopping.restore_prior_model(model)
            break
        
        for name, value in val_metrics.items():
            mlflow.log_metric(f"val_{name}", value, step = epoch)
            
        scheduler.step(val_metrics["loss"])
        
        metric_display = " | ".join(f"{k.upper()}: {v:.5f}" for k, v in train_metrics.items())
        progress_bar.set_description(
            f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.5f} | {metric_display}"
        )
        
    best_threshold, best_f1 = find_best_threshold(model, val_loader, device)
    mlflow.log_metric("best_threshold", best_threshold)
    mlflow.log_metric("best_f1", best_f1)