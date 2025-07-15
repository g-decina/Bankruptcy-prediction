import pandas as pd
from dataclasses import dataclass
from src.data.time_series import MacroTimeSeries
from src.data.tensor_factory import TensorFactory

@dataclass
class SeriesDataset:
    financial_df: pd.DataFrame
    macro_series: list[MacroTimeSeries]
    years: list[str]
    label_col: str|None = None
    
    def to_tensors(self, factory: TensorFactory, fin_scale=True, macro_scale=False):
        X = factory.financial_to_tensor(self.financial_df, fin_scale)
        M = factory.macro_to_tensor(self.macro_series, macro_scale)
        y = (self.financial_df[self.label_col].values if self.label_col else None)
        return X, M, y