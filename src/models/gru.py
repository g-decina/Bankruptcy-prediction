import torch
import torch.nn as nn
from src.utils.utils import normalize_logits

import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(
        self, 
        firm_input_size, 
        macro_input_size, 
        hidden_size, 
        output_size, 
        num_layers = 2, 
        dropout = 0.2
    ):
        super().__init__()
        self.type = "gru"
        
        # ---- For company-level data ----
        self.firm_gru = nn.GRU(
            input_size = firm_input_size, 
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout if num_layers > 1 else 0.0
        )

        # ---- For macro-level data ----
        self.macro_gru = nn.GRU(
            input_size = macro_input_size, 
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout if num_layers > 1 else 0.0
        )

        # ---- Joining the two sub-networks ----
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

        self.firm_bn = nn.LayerNorm(hidden_size)
        self.macro_bn = nn.LayerNorm(hidden_size)

        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
                
    def forward(self, firm, macro):
        firm = firm.unsqueeze(1) if firm.dim() == 2 else firm
        macro = macro.unsqueeze(1) if macro.dim() == 2 else macro

        firm_out, _ = self.firm_gru(firm)
        macro_out, _ = self.macro_gru(macro)

        firm_norm = self.firm_bn(firm_out)
        macro_norm = self.macro_bn(macro_out)

        combined = torch.cat([firm_norm, macro_norm], dim=1)
        combined_pooled = combined.mean(dim=1)
        out = self.fc(combined_pooled)
        return out
    
    def predict(self, firm, macro, threshold: float = 0.5) -> dict:
        self.eval()
        with torch.inference_mode():
            logits = self(firm.unsqueeze(0), macro.unsqueeze(0))
            prob = torch.sigmoid(logits).item()
            return {
                "probability": prob,
                "prediction": int(prob >= threshold)
            }

class EnsembleGRU(nn.Module):
    def __init__(
        self, 
        models: list[GRUModel],
        hidden_sizes: list[int],
        threshold: float=0.5,
        dropout: float=0.3
    ):
        super().__init__()
        self.models = models
        for model in self.models:
            model.eval()
        
        self.input_size = len(self.models)
        self.n_layers = len(hidden_sizes)
        
        self.threshold = threshold
        
        # self.attn = Attention(input_dim = self.input_size)
        self.mlp = self._construct_mlp(hidden_sizes, dropout)
        self._init_weights()
    
    def _construct_mlp(
        self, 
        hidden_size: list[int], 
        dropout: float, 
        activation: type[nn.Module]=nn.GELU
    ) -> nn.Sequential:
        in_dim = self.input_size
        layers=[]
        
        for _, hidden_dim in enumerate(hidden_size):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, 1))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
    
    def forward(self, firm, macro):
        firm = firm.unsqueeze(1) if firm.dim() == 2 else firm
        macro = macro.unsqueeze(1) if macro.dim() == 2 else macro
        
        with torch.inference_mode():
            raw_logits = [model(firm, macro) for model in self.models]
            
        logits_tensor = torch.stack(raw_logits, dim=1).squeeze(-1) # (B, n_models, 1)
        normalized_logits = normalize_logits(logits_tensor)
        
        # context, _ = self.attn(logits_tensor)
        return self.mlp(normalized_logits)
    
    def predict_one(
        self,
        firm,
        macro
    ):
        logits = self.forward(firm, macro)
        prob = torch.sigmoid(logits).item()
        return {
            "probability": prob,
            "prediction": int(prob >= self.threshold)
        }