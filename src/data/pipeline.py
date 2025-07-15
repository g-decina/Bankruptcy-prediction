import mlflow.tracking
import pandas as pd
import numpy as np
import json
import torch
import logging
import mlflow

from pathlib import Path
from src.datasets.series_dataset import SeriesDataset
from src.data.loaders import CompanyDataLoader, MacroDataLoader
from src.data.tensor_factory import TensorFactory
from src.data.time_series import MacroTimeSeries
from src.data.feature_engineer import FeatureEngineer

logger = logging.getLogger(__name__)

class IngestionPipeline:
    def __init__(self, 
            company_path: Path, 
            macro_paths: list[Path],
            company_col: str, 
            bankruptcy_col: str,
            mode: str="train",
            revenue_cap: int=3000
        ):
        self.company_col=company_col
        self.bankruptcy_col=bankruptcy_col
        self.mode=mode
        
        self.company_loader=CompanyDataLoader(
            company_path, sheet_name="Results", revenue_cap=revenue_cap
        )
        self.macro_loader=MacroDataLoader(macro_paths)
        
        self.feature_engineer=None
        self.tensor_factory=None
        self.series_dataset=None
        
    def run(self):
        company_df = self.company_loader.load(
            company_col=self.company_col,
            bankruptcy_col=self.bankruptcy_col,
            mode=self.mode
        )
        
        macro_df= self.macro_loader.load()
        
        years=self.company_loader.years
        self.tensor_factory=TensorFactory(years=years)
        
        macro_series=[
            MacroTimeSeries(macro_df.iloc[:,i], years) for i in range(macro_df.shape[1])
        ]
        
        label_col=f"bankrupt_{years[-1]}" if self.mode == "train" else None
        self.feature_engineer=FeatureEngineer(years=years, label_col=label_col)
        
        if self.mode == "Train":
            company_df=self.feature_engineer.fit_transform(company_df)
        else:
            company_df=self.feature_engineer.transform(company_df)
            
        self.series_dataset=SeriesDataset(
            financial_df=company_df, macro_series=macro_series, years=years, label_col=label_col
        )
        
    def get_tensors(self, fin_scale: bool=True, macro_scale: bool=False):
        assert self.series_dataset is not None, "Pipeline not yet run. Call run() before exporting tensors."
        return self.series_dataset.to_tensors(
            factory=self.tensor_factory,
            fin_scale=fin_scale,
            macro_scale=macro_scale
        )