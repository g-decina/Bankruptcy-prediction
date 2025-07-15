import torch
from types import SimpleNamespace

from src.models.gru import GRUModel
from src.datasets.series_dataset import SeriesDataset
from train_entry import TrainConfig

def test_model_forward():
    # Minimal config for testing
    config = SimpleNamespace(
        data_path="demo_data.xlsx",
        target_year=2025,
        micro_window=3,
        macro_window=5,
        prediction_years=2,
        batch_size=2,
        hidden_dim=32,
        dropout=0.2,
        learning_rate=0.001,
        n_epochs=1,
        seed=42
    )

    # Initialize handler
    handler = TrainConfig(config)
    X_micro, X_macro, y, _, _ = handler.prepare_datasets()

    # Use only the first batch for quick test
    dataset = SeriesDataset(X_micro, X_macro, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size)

    # Model instantiation
    input_dim_micro = X_micro.shape[2]
    input_dim_macro = X_macro.shape[2]
    model = GRUModel(input_dim_micro, input_dim_macro, hidden_dim=config.hidden_dim)

    # Sanity check on one batch
    for x_micro_batch, x_macro_batch, y_batch in loader:
        out = model(x_micro_batch, x_macro_batch)
        assert out.shape == (config.batch_size, 1), f"Unexpected output shape: {out.shape}"
        assert not torch.isnan(out).any(), "Model output contains NaNs"
        break

    print("✅ Forward pass test successful.")

if __name__ == "__main__":
    test_model_forward()
