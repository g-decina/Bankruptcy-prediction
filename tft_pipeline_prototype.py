import torch
import torch.nn as nn
import datetime
import logging
import mlflow
import yaml

from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from tqdm import tqdm
from pathlib import Path
from src.data.pipeline import IngestionPipeline
from src.datasets.dual_input import DualInputSequenceDataset
from src.models.tft import TFTModel
from src.utils.utils import CustomReduceLROnPlateau, TrainConfig, collate_with_macro

def load_yaml_file(path):
    with open(path) as stream:
        try:
            config_dict=yaml.safe_load(stream)
            return config_dict
        except yaml.YAMLError as e:
            TypeError(f"Config file could not be loaded: {e}")


def train_tft(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    metrics: MetricCollection,
    criterion: nn.Module = nn.MSELoss(),
    optimizer_cls=torch.optim.Adam,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "mps" 
        if torch.mps.is_available() else "cpu",
    early_stopping_patience: int = 10,
    scheduler_cls=None,
    scheduler_kwargs=None,
    log_fn=None,
):
    
    model = model.to(device)
    optimizer=optimizer_cls(model.parameters(), lr=lr)
    scheduler=scheduler_cls(optimizer, **scheduler_kwargs) if scheduler_cls else None
    
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} / {num_epochs} — Training"):
            firm_x, macro_x, y = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            y_hat = model(firm_x, macro_x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
        train_loss = sum(train_losses) / len(train_losses)
        computed_metrics = {
            name: metric.compute().item() for name, metric in metrics.items()
        }
        computed_metrics["loss"] = train_loss
        for name, value in metrics.items():
            mlflow.log_metric(f"train_{name}", value, step = epoch)
        history["train_loss"].append(train_loss)
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} / {num_epochs} — Validation"):
                firm_x, macro_x, y = [b.to(device) for b in batch]
                y_hat = model(firm_x, macro_x)
                val_loss = criterion(y_hat, y)
                val_losses.append(val_loss.item())
        
        val_loss = sum(val_losses) / len(val_losses)
        computed_metrics = {
            name: metric.compute().item() for name, metric in metrics.items()
        }
        for name, value in metrics.items():
            mlflow.log_metric(f"val_{name}", value, step = epoch)
        history["val_loss"].append(val_loss)
        
        if log_fn:
            log_fn(epoch=epoch, train_loss=train_loss, val_loss=val_loss)
        
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_tft_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

        if scheduler:
            scheduler.step(val_loss)

    model_name = f"model_{datetime.datetime.now()}"
    mlflow.pytorch.log_model(model, model_name)
    torch.save(obj = model.state_dict(), f = f"models/{model_name}.pth")
    print(f"Model saved: {model_name}")
    
    return model, history


def main():
    config_dict = load_yaml_file("config/model_config.yml")
    cfg = TrainConfig(**config_dict)

    company_path = Path(cfg.firm_data)
    macro_paths = [Path(path) for path in cfg.macro_data]
    bankruptcy_col = str(cfg.bankruptcy_col)
    company_col=str(cfg.company_col)
    revenue_cap=int(cfg.revenue_cap)
    metrics=cfg.get_metrics().to(cfg.device)
    device=str(cfg.device)
    num_layers=int(cfg.num_classes)
    hidden_size=int(cfg.hidden_size)
    output_size=1
    epochs=int(cfg.epochs)
    lr=float(cfg.lr)
    train_fract=float(cfg.train_fract)
    dropout=int(cfg.dropout)
    scheduler_factor=float(cfg.scheduler_factor)
    scheduler_patience=int(cfg.scheduler_patience)
    decay_ih=float(cfg.decay_ih)
    decay_hh=float(cfg.decay_hh)
    decay_other=float(cfg.decay_other)
    seed=int(cfg.seed)
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    
    ingestion = IngestionPipeline(
        company_path=company_path,
        macro_paths=macro_paths,
        company_col=company_col,
        bankruptcy_col=bankruptcy_col,
        revenue_cap=revenue_cap
    )
    
    ingestion.run()
    
    X, M, y = ingestion.get_tensors()
    
    dataset = DualInputSequenceDataset(
        firm_tensor = X,
        macro_tensor = M,
        labels = y
    )
    
    train_ds, val_ds, seed = dataset.stratified_split(train_fract)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_with_macro)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_with_macro)

    logger.info(f"Device: {device}")
    
    firm_input_size, macro_input_size = dataset.input_dims()
    
    optimizer = Adam([
        {'params': decay_ih, 'weight_decay': decay_ih},
        {'params': decay_hh, 'weight_decay': decay_hh},
        {'params': decay_other, 'weight_decay': decay_other},
    ], lr=lr)
    
    scheduler=CustomReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=0.0
        )
    
    pos_weight = dataset.pos_weight()
    loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight)
    
    mlflow.set_tracking_uri('http://127.0.0.1:8080')
    mlflow.set_experiment('bankruptcy-predictions')
    
    # Logging hyperparameters
    mlflow.log_param("hidden_size", hidden_size)
    mlflow.log_param("output_size", output_size)
    mlflow.log_param("num_layers", num_layers)
    mlflow.log_param("dropout", dropout)
    mlflow.log_param("lr", lr)
    
    with mlflow.start_run():
        mlflow.log_param("seed", seed)
        model = TFTModel(
            static_input_dim=X.size,
            encoder_input_dims=M.size,
            decoder_input_dims=M.size,
            hidden_dim=hidden_size,
            attention_heads=4,
            dropout=dropout
        )
        
        train_tft(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=epochs,
            criterion=loss_fn,
            optimizer_cls=optimizer,
            lr=lr,
            scheduler_cls=scheduler,
            metrics=metrics
        )