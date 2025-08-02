
import numpy as np
import pandas as pd
import torch
import logging

from sklearn.preprocessing import RobustScaler

from src.data.time_series import MacroTimeSeries

logger = logging.getLogger(__name__)

class TensorFactory:
    def __init__(self, years: list[str]):
        self.years = years
        
    def financial_to_tensor(self, df: pd.DataFrame, scale: bool = True) -> tuple[np.ndarray]:
        """
        Reshape cleaned DataFrame into RNN-compatible tensor:
        (samples, timesteps, features)

        Args:
            df (pd.DataFrame): Output from _clean_company_dataset()

        Returns:
            tuple: (X, y) where:
                - X is np.ndarray of shape (n_samples, 3, 4)
                - y is np.ndarray of shape (n_samples,)
        """        
        logger.info("Converting financial series to tensors...")

        # Extract per-year features
        df = df.copy()
        
        feature_blocks = []
        for y in self.years:
            features = df[[
                f"revenue_{y}", f"ebt_{y}", f"sheq_{y}", f"cf_{y}", f"net_marg_{y}",
                f"roa_{y}", f"roe_{y}", f"cf_marg_{y}", f"cf_roa_{y}", f"eq_ratio_{y}",
                f"eq_ratio_{y}", f"asset_tov_{y}", f"rev_p_unit_ats_{y}"]].to_numpy()
            feature_blocks.append(features)
        
        # Stack: [samples, timesteps=3, features=4]
        X = np.stack(feature_blocks, axis=1)
        
        if scale:
            logger.info("Scaling financial data with RobustScaler...")
            X_scaled = np.empty_like(X)
            
            _, n_timesteps, n_features = X.shape
            scalers = [[None for _ in range(n_timesteps)] for _ in range(n_features)]
            for t in range(n_timesteps):
                for i in range(n_features):
                    scaler=RobustScaler()
                    feature_t = X[:, t, i].reshape(-1, 1)
                    X_scaled[:, t, i]=scaler.fit_transform(feature_t).flatten()
                    scalers[i][t]=(scaler) 
                    # TO ADD: return scalers for future transformation
            
            X = X_scaled
            
        logger.info(f"Financial data tensor shape: {X.shape}")

        return torch.tensor(X, dtype=torch.float32)
    
    def macro_to_tensor(self, macro_series: list[MacroTimeSeries], scale: bool = False):
        logger.info("Converting macroeconomic series to tensors...")
        
        stacked=torch.stack([s.to_tensor().float() for s in macro_series])
        
        if scale:
            logger.info("Scaling macro data with RobustScaler...")
            scaled = []
            for s in stacked:
                scaled_tensor = torch.tensor(
                    RobustScaler().fit_transform(s.numpy().reshape(-1, 1)).flatten()
                )
                scaled.append(scaled_tensor)
            stacked = torch.stack(scaled)
            logger.info(f"Shaped macro data tensor: {torch.stack(scaled).shape}")
        
        if stacked.shape[1] < 12:
            raise ValueError(f"Expected at least 12 future periods, got shape: {stacked.shape}")
        
        past=stacked[:, :-12]
        future=stacked[:, -12:]
        
        logger.info(f"Macro data tensor shape: {past.shape} (past), {future.shape} (future)")
        
        return past, future