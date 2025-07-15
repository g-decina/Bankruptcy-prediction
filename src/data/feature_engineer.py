
import numpy as np
import pandas as pd
import json
import logging
import mlflow

from pathlib import Path

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, years: list[str], label_col: str|None = None):
        self.years = years
        self.label_col = label_col
        self.expected_columns = None
    
    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generates derived financial features."""
        df = df.copy()
        years = sorted({col[-4:] for col in df.columns if "revenue_" in col})
        
        logger.info("Engineering features...")
        # ---- Growth features ---- 
        if len(years) >= 3:
            y1, y2, y3 = years[-3], years[-2], years[-1]
            try:
                df[f"revenue_growth_{y2}_{y3}"] = (
                    df[f"revenue_{y3}"] - df[f"revenue_{y2}"]
                    ) / df[f"revenue_{y2}"]
                df[f"revenue_growth_{y1}_{y2}"] = (
                    df[f"revenue_{y2}"] - df[f"revenue_{y1}"]
                    ) / df[f"revenue_{y1}"]
            except KeyError as e:
                logger.warning(f"Missing column during growth computation: {e}")
        
        revenue_cols = [f"revenue_{y}" for y in years if f"revenue_{y}" in df.columns]
        ebt_cols = [f"ebt_{y}" for y in years if f"ebt_{y}" in df.columns]
        cf_cols = [f"cf_{y}" for y in years if f"cf_{y}" in df.columns]
        
        # ---- Ratio features ---- 
        try:
            df["ebt_margin_avg"] = df[ebt_cols].mean(axis = 1) / df[revenue_cols].mean(axis = 1)
            df["cf_margin_avg"] = df[cf_cols].mean(axis = 1) / df[revenue_cols].mean(axis = 1)
        except ZeroDivisionError:
            logger.warning("Zero division occured during margin computation")

        # ---- Volatility features ----
        if len(revenue_cols) >= 2:
            df["revenue_volatility"] = df[revenue_cols].std(axis=1)
        if len(ebt_cols) >= 2:
            df["ebt_volatility"] = df[ebt_cols].std(axis=1)
        if len(cf_cols) >= 2:
            df["cf_volatility"] = df[cf_cols].std(axis=1)
        
        # ---- Cleanup ---- 
        
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        return df
        
    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies transformations and stores expected columns."""
        logger.info("Fitting feature engineer...")
        df = self._engineer(df.copy())
        self.expected_columns = df.columns.tolist()
        
        missing = [col for col in self.expected_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing expected columns at inference: {missing}")
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Transforming new data with fitted features...")
        
        df = df.copy()
        label_data = None
        
        if self.label_col and self.label_col in df.columns:
            label_data = df[self.label_col]
            
        df = self._engineer(df)
        
        if self.expected_columns:
            missing = [col for col in self.expected_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Missing expected columns at inference: {missing}")
        
        if label_data is not None:
            df[self.label_col] = label_data
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies transformations and reindexes to expected columns."""
        df = df.copy()
        
        return self.fit(df).transform(df)
    
    def save(self, path: str):
        path = Path(path)
        with open(path, "x") as f:
            json.dump(self.expected_columns, f)
            
    def load(self, path: str):
        path = Path(path)
        with open(path, "r") as f:
            self.expected_columns = json.load(f)
            
    def save_feature_schema(self, artifact_path="features/expected_columns.json"):
        tmp_path = Path("temp_expected_columns.json")
        with open(tmp_path, "w") as f:
            json.dump(self.expected_columns, f)
        mlflow.log_artifact(tmp_path, artifact_path)
        tmp_path.unlink()
    
    def load_feature_schema_from_mlflow(self, run_id: str, artifact_path: str):
        client = mlflow.tracking.MlflowClient()
        local_path = client.download_artifacts(run_id, artifact_path)
        with open(local_path, "r") as f:
            self.expected_columns=json.load(f)