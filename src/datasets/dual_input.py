import numpy as np
import numpy.random as rand
import torch
import logging

from torch.utils.data import Dataset, random_split
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Optional

logger = logging.getLogger(__name__)

class DualInputSequenceDataset(Dataset):
    def __init__(
        self, 
        firm_tensor: torch.Tensor, 
        macro_past_tensor: torch.Tensor, 
        macro_future_tensor: torch.Tensor, 
        labels: torch.Tensor
    ):
        """
        Custom dataset class made to join firm-level and macro-level data.
        
        Dims:
            firm_tensor: (N, T, F_firm)
            macro_past_tensor (_type_): (T, F_macro - 12) — shared across whole batch
            macro_future_tensor (_type_): (T, 12) — shared across whole batch
            labels (_type_): (N, 1)
        """
        # ---- numpy -> torch conversion if needed ----
        if isinstance(firm_tensor, np.ndarray):
            firm_tensor = torch.tensor(firm_tensor, dtype=torch.float32)
        if isinstance(macro_past_tensor, np.ndarray):
            macro_past_tensor = torch.tensor(macro_past_tensor, dtype=torch.float32)
        if isinstance(macro_future_tensor, np.ndarray):
            macro_future_tensor = torch.tensor(macro_future_tensor, dtype=torch.float32)
        if isinstance(labels, np.ndarray):
            labels = torch.tensor(labels, dtype=torch.float32)
        
        # ---- ensuring 3D (N, T, F) for firm_tensor ----
        if firm_tensor.dim() == 2:
            firm_tensor = firm_tensor.unsqueeze(-1)
        elif firm_tensor.dim() == 3:
            pass
        else:
            raise ValueError(f"Firm tensor must be 2D, or 3D. Got {firm_tensor.dim()}D tensor.")
            
        self.firm_tensor = firm_tensor
        self.macro_past_tensor = macro_past_tensor
        self.macro_future_tensor = macro_future_tensor
        self.labels = labels.view(-1, 1)
        
    def __len__(self):
        return self.firm_tensor.shape[0]
    
    def __getitem__(self, idx):
        return {
            "firm_seq": self.firm_tensor[idx],
            "macro_past": self.macro_past_tensor,
            "macro_future": self.macro_future_tensor,
            "label": self.labels[idx]
        }
        
    def input_dims(self):
        return self.firm_tensor.shape, self.macro_past_tensor.shape, self.macro_future_tensor.shape
    
    def label_distribution(self):
        num_pos = (self.labels == 1).sum().item()
        num_neg = (self.labels == 0).sum().item()
        return num_pos, num_neg
        
    def pos_weight(self):
        num_pos, num_neg = self.label_distribution()
        return torch.tensor(num_neg / num_pos, dtype = torch.float32)
    
    def stratified_split(self, train_fract: float = 0.8, seed: Optional[int] = None):
        firm_tensor_cpu = self.firm_tensor.cpu().numpy()
        labels_flat_cpu = self.labels.cpu().numpy().squeeze()
        
        if seed is None:
            seed = rand.randint(1, 1_000_000)
            
        splitter = StratifiedShuffleSplit(
            n_splits=1, 
            train_size=train_fract, 
            random_state=seed
        )
        
        train_idx, val_idx = next(splitter.split(X = firm_tensor_cpu, y = labels_flat_cpu))
        
        train_dataset = DualInputSequenceDataset(
            self.firm_tensor[train_idx],
            self.macro_past_tensor,
            self.macro_future_tensor,
            self.labels[train_idx]
        )
        
        val_dataset = DualInputSequenceDataset(
            self.firm_tensor[val_idx],
            self.macro_past_tensor,
            self.macro_future_tensor,
            self.labels[val_idx]
        )
        
        return train_dataset, val_dataset, seed
    
    def random_split(self, train_fract: float = 0.8, seed: Optional[int] = None):
        if seed is None:
            seed = rand.randint(1, 1_000_000)
        
        torch.manual(seed)
        train_size = int(train_fract * len(self))
        val_size = len(self) - train_size
        
        return random_split(self.firm_tensor, [train_size, val_size]), seed

    def to_device(self, device: torch.device):
        """
        Moves all of the tensors to the specified device.

        Args:
            device (torch.device): The target device ('cpu', 'cuda' or 'mps').
        """
        self.firm_tensor = self.firm_tensor.to(device)
        self.macro_past_tensor = self.macro_past_tensor.to(device)
        self.macro_future_tensor = self.macro_future_tensor.to(device)
        self.labels = self.labels.to(device)
        
        print(f"Data sent to device: {device}")