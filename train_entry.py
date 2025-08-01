import torch
import mlflow
import datetime
import logging
import yaml

from dataclasses import dataclass, field
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy, BinaryAUROC, BinaryF1Score, BinaryMatthewsCorrCoef,
    MulticlassAccuracy, MulticlassAUROC, MulticlassF1Score)
from pathlib import Path

from src.datasets.dual_input import DualInputSequenceDataset
from src.models.gru import GRUModel
from src.data.pipeline import IngestionPipeline
from train.gru import train_model
from src.utils.utils import CustomReduceLROnPlateau, collate_with_macro

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def load_yaml_file(path):
    with open(path) as stream:
        try:
            config_dict=yaml.safe_load(stream)
            return config_dict
        except yaml.YAMLError as e:
            TypeError(f"Config file could not be loaded: {e}")
    

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
    scheduler_factor: float=0.85
    scheduler_patience: int = 50
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
    
def train_model_from_config(cfg: TrainConfig) -> GRUModel:
    """Main training function"""
    return train(
        company_path = Path(cfg.firm_data),
        macro_paths = [Path(path) for path in cfg.macro_data],
        bankruptcy_col = str(cfg.bankruptcy_col),
        company_col=str(cfg.company_col),
        revenue_cap=int(cfg.revenue_cap),
        metrics=cfg.get_metrics().to(cfg.device),
        device=str(cfg.device),
        num_layers=int(cfg.num_classes),
        hidden_size=int(cfg.hidden_size),
        output_size=1,
        epochs=int(cfg.epochs),
        lr=float(cfg.lr),
        train_fract=float(cfg.train_fract),
        dropout=int(cfg.dropout),
        scheduler_factor=float(cfg.scheduler_factor),
        scheduler_patience=int(cfg.scheduler_patience),
        decay_ih=float(cfg.decay_ih),
        decay_hh=float(cfg.decay_hh),
        decay_other=float(cfg.decay_other),
        seed=int(cfg.seed)
    )

def train(
    company_path: str,
    macro_paths: list[str],
    bankruptcy_col: str,
    company_col: str,
    revenue_cap: int,
    metrics: list[Metric],
    seed: int,
    num_layers: int = 2,
    hidden_size: int = 64,
    output_size: int = 1,
    epochs: int = 50,
    lr: float = 1e-3,
    train_fract: float = 0.8,
    dropout: float = 0.2,
    scheduler_factor: float = 0.85,
    scheduler_patience: int = 50,
    min_lr: float = 0.0,
    decay_ih:float = 1e-5,
    decay_hh:float = 1e-5,
    decay_other:float = 1e-5,
    device: str="cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
):  
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
    
    metrics.to(device)
    train_ds = train_ds.to_device(device)
    val_ds = val_ds.to_device(device)
    
    firm_input_size, macro_input_size = dataset.input_dims()
    
    mlflow.set_tracking_uri('http://127.0.0.1:8080')
    mlflow.set_experiment('bankruptcy-predictions')
    
    with mlflow.start_run():
        mlflow.log_param("seed", seed)
        model = GRUModel(
            firm_input_size=firm_input_size,
            macro_input_size=macro_input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        model = model.to(device)
        
        pos_weight = dataset.pos_weight()
        loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Logging hyperparameters
        mlflow.log_param("hidden_size", hidden_size)
        mlflow.log_param("output_size", output_size)
        mlflow.log_param("num_layers", num_layers)
        mlflow.log_param("dropout", dropout)
        mlflow.log_param("lr", lr)

        ih_params = []
        hh_params = []
        other_params = []

        for name, param in model.named_parameters():
            if 'weight_ih' in name:
                ih_params.append(param)
            elif 'weight_hh' in name:
                hh_params.append(param)
            else:
                other_params.append(param)
        
        optimizer = Adam([
                {'params': ih_params, 'weight_decay': decay_ih},
                {'params': hh_params, 'weight_decay': decay_hh},
                {'params': other_params, 'weight_decay': decay_other},
            ], lr=lr
        )
        scheduler=CustomReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=min_lr
        )
        
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=epochs,
            metrics=metrics
        )
        
        model_name = f"model_{datetime.datetime.now()}"
        mlflow.pytorch.log_model(model, model_name)
        torch.save(obj = model.state_dict(), f = f"models/{model_name}.pth")
        print(f"Model saved: {model_name}")
    
    return model