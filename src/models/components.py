import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor=None):
        scores = self.attn(x)
        
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
            
        weights = torch.softmax(scores, dim=1)  # (batch_size, seq_len)
        context = torch.sum(weights.unsqueeze(-1) * x, dim = 1)
        return context, weights

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, gru_outputs):
        scores = self.attn(gru_outputs).squeeze(-1)
        weights = torch.softmax(scores, dim=1)  # (batch_size, seq_len)
        context = torch.bmm(weights.unsqueeze(1), gru_outputs)
        return context.squeeze(1)

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, dropout=dropout
        )
        self.norm=nn.LayerNorm(hidden_size)
        self.dropout=nn.Dropout(dropout)
        
    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.norm(x + self.dropout(attn_output))
        pooled = x.mean(dim=1)
        return pooled

class GatedResidualNetwork(nn.Module):
    """Basic building block of TFT"""
    def __init__(self, input_dim, hidden_dim, context_dim=None, output_dim=None, dropout=0.1):
        super().__init__()
        self.output_dim = output_dim or input_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        
        self.linear_input = nn.Linear(input_dim, hidden_dim)
        self.linear_context = nn.Linear(context_dim, hidden_dim) if context_dim else None
        
        self.ELU = nn.ELU()
        self.linear_hidden = nn.Linear(self.hidden_dim, self.output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.gate_linear = nn.Linear(self.output_dim, self.output_dim)
        self.layer_norm = nn.LayerNorm(self.output_dim)
        
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
        
    def forward(self, x, context=None):
        residual = self.skip(x) if self.skip else x
        x_in = self.linear_input(x)
        
        if context is not None and self.linear_context is not None:
            context = self.linear_context(context)
            
            if context.dim() == 2:
                context = context.unsqueeze(1).expand(-1, x_in.size(1), -1) # (B, T, hidden_dim)
            elif context.shape[1] != x_in.shape[1]:
                raise ValueError(f"Context time dim {context.shape[1]} does not match input time dim {x_in.shape[1]}")
            
            x_in = x_in + context
        
        x = self.ELU(x_in)
        x = self.dropout(x)
        x = self.linear_hidden(x)
        gate = torch.sigmoid(self.gate_linear(x))
        x = gate * x
        x = x + residual
        return self.layer_norm(x)
    
class VariableSelectionNetwork(nn.Module):
    """Computes features importance and projects selected features forward"""
    def __init__(self, input_dim, num_inputs, hidden_dim, context_dim=None, dropout=0.1):
        super().__init__()
        self.num_inputs = num_inputs
        self.input_dim = input_dim
        
        self.flattened_grn = GatedResidualNetwork(
            input_dim=num_inputs * input_dim, 
            hidden_dim=hidden_dim, 
            context_dim=context_dim,
            output_dim=num_inputs, 
            dropout=dropout
        )
        self.vars_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_dim=input_dim, 
                hidden_dim=hidden_dim,
                context_dim=context_dim, 
                output_dim=input_dim,
                dropout=dropout)
            for _ in range(num_inputs)
        ])
        
    def forward(self, embedding_list, context=None): 
        combined = torch.cat(embedding_list, dim=-1) # (B, T, D_total)
        raw_weights = self.flattened_grn(combined, context)
        
        sparse_weights=torch.softmax(raw_weights, dim=-1)
        sparse = sparse_weights.unsqueeze(2) # (B, T, 1, n)
        transformed = [
            grn(embedding_list[i], context) # (B, T, D)
            for i, grn in enumerate(self.vars_grns)
        ] 
        
        stacked = torch.stack(transformed, dim=-1) # (B, T, D, n)
        output = torch.sum(stacked * sparse, dim=-1) # (B, T, D)
        
        return output, sparse # output: context vector
        
class StaticCovariateEncoder(nn.Module):
    """Encodes static metadata and provides context vectors to downstream modules"""
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        self.context_enrichment = GatedResidualNetwork(
            input_dim=hidden_dim, hidden_dim=hidden_dim, context_dim=hidden_dim, 
            output_dim=hidden_dim, dropout=dropout
        )
        self.selection_context = GatedResidualNetwork(
            input_dim=hidden_dim, hidden_dim=hidden_dim, context_dim=hidden_dim, 
            output_dim=hidden_dim, dropout=dropout
        )
        self.state_context = GatedResidualNetwork(
            input_dim=hidden_dim, hidden_dim=hidden_dim, context_dim=hidden_dim, 
            output_dim=hidden_dim, dropout=dropout
        )
    
    def forward(self, static_inputs):
        h = self.input_projection(static_inputs)
        
        sel_ctx=self.selection_context(h)
        temp_ctx=self.context_enrichment(h)
        state_ctx=self.state_context(h)
        
        return sel_ctx, temp_ctx, state_ctx

class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead=nn.MultiheadAttention(
            embed_dim=input_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.projection=nn.Linear(input_dim, input_dim)
        
    def forward(self, x, attn_mask=None): # (B, T, D)
        attn_out, attn_w = self.multihead(
            x, x, x,
            attn_mask=attn_mask,
            need_weights=True, 
            average_attn_weights=False
        )
        B, H, T, _ = attn_w.shape
        attn_w = attn_w.reshape(B * H, T, T)
        attn_w = attn_w / attn_w.sum(dim=-1, keepdim=True)
        
        output=self.projection(attn_out)

        return output, attn_w.detach()