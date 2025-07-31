import torch
import torch.nn as nn

from src.models.components import *

class TFTModel(nn.Module):
    def __init__(
        self,
        static_input_dim,
        company_input_dim,
        macro_input_dim,
        decoder_input_dim,
        hidden_dim=64,
        attention_heads=4,
        dropout=0.1
    ):
        super().__init__()
        
        assert hidden_dim % attention_heads == 0
        self.hidden_dim = hidden_dim
        
        if static_input_dim > 0:
            self.static_encoder = StaticCovariateEncoder(
                input_dim=static_input_dim, 
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        else:
            self.static_encoder = None
        
        self.company_projections = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(company_input_dim)
        ])
        self.macro_projections = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(macro_input_dim)
        ])
        self.decoder_input_projections = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(decoder_input_dim)
        ])
        self.encoder_varsel = VariableSelectionNetwork(
            input_dim=hidden_dim, 
            num_inputs=company_input_dim + macro_input_dim,
            hidden_dim=hidden_dim,
            context_dim=hidden_dim,
            dropout=dropout
        )
        self.decoder_varsel = VariableSelectionNetwork(
            input_dim=hidden_dim, 
            num_inputs=decoder_input_dim,
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
    
    def forward(self, static_inputs, encoder_company_inputs, 
                encoder_macro_inputs, decoder_inputs, encoder_mask=None):
        """
        Parameters:
            static_inputs: (B, D_static)
            encoder_inputs: list of (B, T_encoder, D_i)
            decoder_inputs: list of (B, T_decoder, D_j)
        Returns:
            logits: (B, T_decoder, 1)
        """
        print("encoder_inputs.shape", encoder_macro_inputs.shape, encoder_company_inputs.shape)
        print("decoder_inputs.shape", decoder_inputs.shape)
        
        if encoder_macro_inputs.dim == 2:
            B = encoder_company_inputs.shape[0]
            encoder_macro_inputs = encoder_macro_inputs.unsqueeze(0).expand(B, -1, -1) 
        
        # ---- 0. Project temporal inputs ----
        company_embeds = [
            self.company_projections[i](encoder_company_inputs[..., i].unsqueeze(-1))
            for i in range(encoder_company_inputs.shape[-1])
        ]

        macro_embeds = [
            self.macro_projections[i](encoder_macro_inputs[..., i].unsqueeze(-1))
            for i in range(encoder_macro_inputs.shape[-1])
        ]
        
        encoder_embeds = company_embeds + macro_embeds
        
        decoder_embeds = [
            self.decoder_input_projections[i](decoder_inputs[..., i].unsqueeze(-1))
            for i in range(decoder_inputs.shape[-1])
        ]
        # ---- 1. Static context vectors ----
        if self.static_encoder is not None:
            static_sel_ctx, static_temporal_ctx, static_state_ctx = self.static_encoder(static_inputs)
        else:
            B = encoder_company_inputs.shape[0]
            static_sel_ctx = static_temporal_ctx = static_sel_ctx = torch.zeros(
                B, self.hidden_dim, device=encoder_company_inputs.device
            )
        
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