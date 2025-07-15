import torch
import pytest
from torch import nn

from typing import List

# Import the model components from the user's code
from types import SimpleNamespace

# Dummy instantiations for components
from torch.nn import functional as F

from src.models.components import (StaticCovariateEncoder, 
    GatedResidualNetwork, VariableSelectionNetwork, TemporalAttention,
    InterpretableMultiHeadAttention)

# -- Test StaticCovariateEncoder --

def test_static_covariate_encoder():
    batch_size, input_dim, hidden_dim = 4, 5, 16
    encoder = StaticCovariateEncoder(input_dim, hidden_dim)
    x = torch.randn(batch_size, input_dim)
    var_sel, temporal_ctx, state_ctx = encoder(x)
    assert var_sel.shape == (batch_size, hidden_dim)
    assert temporal_ctx.shape == (batch_size, hidden_dim)
    assert state_ctx.shape == (batch_size, hidden_dim)

# -- Test GatedResidualNetwork --

def test_gated_residual_network():
    batch_size, seq_len, input_dim, hidden_dim = 4, 10, 8, 16
    x = torch.randn(batch_size, seq_len, input_dim)
    context = torch.randn(batch_size, hidden_dim)
    grn = GatedResidualNetwork(input_dim, hidden_dim, context_dim=hidden_dim, output_dim=hidden_dim)
    out = grn(x, context)
    assert out.shape == (batch_size, seq_len, hidden_dim)

# -- Test VariableSelectionNetwork --

def test_variable_selection_network():
    batch_size, T, input_dim, hidden_dim, num_inputs = 4, 10, 8, 16, 3
    inputs = [torch.randn(batch_size, T, input_dim) for _ in range(num_inputs)]
    context = torch.randn(batch_size, T, hidden_dim)
    vsn = VariableSelectionNetwork(input_dim, num_inputs, hidden_dim, context_dim=hidden_dim)
    output, weights = vsn(inputs, context)
    assert output.shape == (batch_size, T, input_dim)
    assert weights.shape == (batch_size, T, 1, num_inputs)

# -- Test TemporalAttention --

def test_temporal_attention():
    batch_size, T, hidden_size = 4, 10, 16
    x = torch.randn(batch_size, T, hidden_size)
    attn = TemporalAttention(hidden_size)
    context = attn(x)
    assert context.shape == (batch_size, hidden_size)
    
def test_interpretable_multihead_attention():
    batch_size = 4
    seq_len = 12
    embed_dim = 16
    num_heads = 2

    # Input tensor
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Initialize the module
    mha = InterpretableMultiHeadAttention(input_dim=embed_dim, num_heads=num_heads)

    # Forward pass
    output, attn_weights = mha(x)

    # Check output shape
    assert output.shape == (batch_size, seq_len, embed_dim), \
        f"Expected output shape {(batch_size, seq_len, embed_dim)}, got {output.shape}"

    # Attention weights come from nn.MultiheadAttention and are shaped (B * H, L, S)
    expected_shape = (batch_size * num_heads, seq_len, seq_len)
    assert attn_weights.shape == expected_shape, \
        f"Expected attn_weights shape {expected_shape}, got {attn_weights.shape}"

    # Values should sum to 1 across the source sequence (softmaxed attention)
    attn_sums = attn_weights.sum(dim=-1)
    assert torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-5), \
        "Each attention row should sum to 1"
