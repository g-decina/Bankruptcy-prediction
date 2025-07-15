import torch
import torch.nn as nn

from src.models.components import (VariableSelectionNetwork, StaticCovariateEncoder,
    InterpretableMultiHeadAttention, GatedResidualNetwork)

class TFTModel(nn.Module):
    def __init__(
        self,
        static_input_dim,
        encoder_input_dims,
        decoder_input_dims,
        hidden_dim=64,
        attention_heads=4,
        dropout=0.1
    ):
        super().__init__()
        
        assert hidden_dim % attention_heads == 0
        
        self.static_encoder = StaticCovariateEncoder(
            input_dim=static_input_dim, 
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        self.encoder_input_projections = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in encoder_input_dims
        ])
        self.decoder_input_projections = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in decoder_input_dims
        ])
        self.encoder_varsel = VariableSelectionNetwork(
            input_dim=hidden_dim, 
            num_inputs=len(encoder_input_dims),
            hidden_dim=hidden_dim,
            context_dim=hidden_dim,
            dropout=dropout
        )
        self.decoder_varsel = VariableSelectionNetwork(
            input_dim=hidden_dim, 
            num_inputs=len(decoder_input_dims),
            hidden_dim=hidden_dim,
            context_dim=hidden_dim,
            dropout=dropout
        )
        self.attention = InterpretableMultiHeadAttention(
            input_dim=hidden_dim,
            num_heads=attention_heads,
            dropout=dropout
        )
        self.position_grn = GatedResidualNetwork(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            context_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout
        )
        self.output_layer=nn.Linear(hidden_dim, 1)
        self.output_dropout = nn.Dropout(dropout)
    
    def forward(self, static_inputs, encoder_inputs, decoder_inputs, encoder_mask=None):
        """
        Parameters:
            static_inputs: (B, D_static)
            encoder_inputs: list of (B, T_encoder, D_i)
            decoder_inputs: list of (B, T_decoder, D_j)
        Returns:
            logits: (B, T_decoder, 1)
        """
        print("encoder_inputs.shape", encoder_inputs.shape)
        print("decoder_inputs.shape", decoder_inputs.shape)
        print("len(self.encoder_input_projections)", len(self.encoder_input_projections))
        encoder_embeds = [
            self.encoder_input_projections[i](encoder_inputs[..., i].unsqueeze(-1))
            for i in range(encoder_inputs.shape[-1])
        ]

        decoder_embeds = [
            self.decoder_input_projections[i](decoder_inputs[..., i].unsqueeze(-1))
            for i in range(decoder_inputs.shape[-1])
        ]
        # ---- 1. Static context vectors ----
        static_sel_ctx, static_temporal_ctx, static_state_ctx = self.static_encoder(static_inputs)
        
        # ---- 2. Variable selection ----
        encoder_context, encoder_weights = self.encoder_varsel(encoder_embeds, context=static_sel_ctx)
        decoder_context, decoder_weights = self.decoder_varsel(decoder_embeds, context=static_sel_ctx)
        
        # ---- 3. Concatenate past and future ----
        full_context = torch.cat([encoder_context, decoder_context], dim = 1) # (B, T_total, D)
        
        # ---- 4. Temporal Self-Attention ----
        enriched, attn_weights = self.attention(full_context, attn_mask=encoder_mask)
        
        # ---- 5. Positional processing and projection ----
        enriched = self.position_grn(enriched, context=static_state_ctx)
        logits = self.output_layer(enriched)
        logits = self.output_dropout(logits)
        
        return logits[:, -decoder_context.shape[1]:], attn_weights