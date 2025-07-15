import torch
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryAveragePrecision, BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryMatthewsCorrCoef
from train_entry import main

def test_main():
    company_data_path = "demo_data.xlsx"
    macro_data_path = [
        "insee/serie_000857179_04042025/valeurs_mensuelles.csv", # Consumer sentiment
        "insee/serie_001656158_04042025/valeurs_trimestrielles.csv", # Total bankruptcies
        "insee/serie_001763782_04042025/valeurs_mensuelles.csv" # Prices
    ]
    bankruptcy_col="Status date"
    company_col="Company name Latin alphabet"
    revenue_cap=12_000
    
    threshold = 0.4

    metrics = [
        BinaryAveragePrecision([threshold]), BinaryF1Score(threshold), BinaryPrecision(threshold), 
        BinaryRecall(threshold), BinaryMatthewsCorrCoef(threshold)
    ]

    model = main(
        company_data_path=company_data_path,
        macro_data_path=macro_data_path,
        bankruptcy_col=bankruptcy_col,
        company_col=company_col,
        metrics=metrics,
        hidden_size = 64,
        num_layers = 2,
        alpha=None,
        gamma=None,
        revenue_cap=revenue_cap,
        lr=5e-3,
        epochs=100,
        scheduler_patience=10,
        scheduler_factor=0.75,
        dropout=0.5,
        decay_hh = 1e-4
    )
    
    return model

for i in range(100):
    model = test_main()