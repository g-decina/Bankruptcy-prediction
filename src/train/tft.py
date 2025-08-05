import mlflow
import logging
import torch
import io
import matplotlib.pyplot as plt
from tqdm import trange

from torchmetrics.classification import (BinaryAccuracy, BinaryF1Score, 
                            BinaryMatthewsCorrCoef, BinaryPrecision, BinaryRecall,
                            BinaryPrecisionRecallCurve)

from torch.nn.utils import clip_grad_norm_
from src.utils.utils import RollingEarlyStopping, find_best_threshold

def train_one_epoch(model, loader, optimizer, loss_fn, device, metrics):
    model.train()
    for metric in metrics.values():
        metric.reset()
        
    total_loss = 0
    for batch in loader:
        firm_seq = batch["firm_seq"].to(device) # (batch, T, F_firm)
        encoder_inputs = batch["macro_past"].to(device)  # (T, F_macro - 12) - shared across batch
        decoder_inputs = batch["macro_future"].to(device) # (T, 12) - shared across batch
        labels = batch["label"].to(device) # (batch, 1)
        
        B, T, _ = firm_seq.shape
        static_inputs = torch.zeros((B, 6, model.static_input_dim), device=device)
        
        optimizer.zero_grad()
        preds, _ = model(firm_seq, encoder_inputs, decoder_inputs, static_inputs)
        
        loss = loss_fn(preds, labels)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        for metric in metrics.values():
            for metric in metrics.values():
                if isinstance(metric, (
                    BinaryF1Score, BinaryMatthewsCorrCoef, BinaryAccuracy,
                    BinaryPrecision, BinaryRecall
                )):
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

def evaluate_one_epoch(model, loader, loss_fn, device, metrics, log_pr_curve: bool=False):
    model.eval()
    for metric in metrics.values():
        metric.reset()
    
    if log_pr_curve:
        pr_curve = BinaryPrecisionRecallCurve().to(device)
    
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            firm_seq = batch["firm_seq"].to(device)
            encoder_inputs = batch["macro_past"].to(device)
            decoder_inputs = batch["macro_future"].to(device)
            labels = batch["label"].to(device)
            
            B, T, _ = firm_seq.shape
            static_inputs = torch.zeros((B, 1, model.static_input_dim), device=device)
            
            preds, _ = model(firm_seq, encoder_inputs, decoder_inputs, static_inputs)
            loss = loss_fn(preds, labels)
            total_loss += loss.item()
            
            labels = labels.to(torch.int64)
            if log_pr_curve:
                pr_curve.update(torch.sigmoid(preds), labels)
            
            for metric in metrics.values():
                if isinstance(metric, (
                    BinaryF1Score, BinaryMatthewsCorrCoef, BinaryAccuracy,
                    BinaryPrecision, BinaryRecall
                )):
                    preds_binary = (preds.sigmoid() > 0.5).int()
                    metric.update(preds_binary, labels)
                else:
                    metric.update(preds, labels)
                
    avg_loss = total_loss / len(loader)
    computed_metrics = {
        name: metric.compute().item() for name, metric in metrics.items()
    }
    computed_metrics["loss"] = avg_loss
    
    if log_pr_curve:
        pr_curve.compute()
        
        plt.figure()
        fig, ax = plt.subplots()
        pr_curve.plot(ax=ax)
        
        mlflow.log_figure(fig, "plots/pr_curve.png")
        
    return computed_metrics

def train_tft(
    model, train_loader, val_loader, loss_fn, optimizer,
    scheduler, stopping_patience, stopping_window, device, epochs, 
    metrics=None
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
            
        val_metrics = evaluate_one_epoch(model, val_loader, loss_fn, 
                                        device, metrics, log_pr_curve=False)
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
            f"Epoch {epoch}/{epochs} | {metric_display}"
        )
        
    best_threshold, best_f1 = find_best_threshold(model, val_loader, device)
    evaluate_one_epoch(model, val_loader, loss_fn, device, metrics, log_pr_curve=True)
    mlflow.log_metric("best_threshold", best_threshold)
    mlflow.log_metric("best_f1", best_f1)