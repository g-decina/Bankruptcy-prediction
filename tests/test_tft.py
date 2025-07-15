import torch
import pytest

from src.models.tft import TFTModel

# ---- Dummy input specs ----
@pytest.mark.parametrize(
    "batch_size, T_encoder, T_decoder, D_static, D_firm, D_macro, hidden_dim", [
    (4, 12, 6, 5, 3, 2, 16),
    (2, 8, 4, 3, 2, 1, 8)
])

def test_tft_forward_pass(batch_size, T_encoder, T_decoder, D_static,
                        D_firm, D_macro, hidden_dim):
    model =  TFTModel(
        static_input_dim=D_static,
        encoder_input_dims=[D_firm, D_macro],
        decoder_input_dims=[D_macro],  # Only macro is known/predicted in future
        hidden_dim=hidden_dim,
        attention_heads=2,
        dropout=0.1
    )
    
    static_inputs = torch.randn(batch_size, D_static)
    encoder_inputs = [
        torch.randn(batch_size, T_encoder, D_firm),
        torch.randn(batch_size, T_encoder, D_macro)
    ]
    
    decoder_inputs = [
        torch.randn(batch_size, T_decoder, D_macro)
    ]
    
    logits, attn_weights = model(static_inputs, encoder_inputs, decoder_inputs)
    
    assert logits.shape == (batch_size, T_decoder, 1), \
        f"Expected logits shape {(batch_size, T_decoder, 1)}, got {logits.shape}"
    
    expected_attn_length = T_encoder + T_decoder
    assert attn_weights.shape[1] == expected_attn_length and \
        attn_weights.shape[2] == expected_attn_length, \
        f"Expected attention length (B, {expected_attn_length}, {expected_attn_length}), \
        got {attn_weights.shape}"
    