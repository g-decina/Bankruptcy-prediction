import numpy as np
import pandas as pd
import torch
import datetime
import logging

from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)

class TimeSeries:
    def __init__(
        self, 
        data: pd.DataFrame
    ):
        self.data = data
        self.predicted = None
    
    @staticmethod
    def average_per_year(df: pd.DataFrame) -> pd.DataFrame:
        """
        Resamples infra-annual forecasts to annual averages.

        Args:
            df (pd.DataFrame): a time series from the forecast_prophet function
            n_periods (int, optional): n of periods in the initial df. Defaults to 12 (monthly data).

        Returns:
            pd.DataFrame: an annual dataframe
        """
        
        df = df.copy()
        if "yhat" not in df.columns or "ds" not in df.columns:
            raise ValueError("Expected columns 'yhat' and 'ds' in the dataframe.")

        df["ds"] = pd.to_datetime(df["ds"])
        df["year"] = df["ds"].dt.year
        
        return df.groupby("year")["yhat"].mean().reset_index()
    
    def compute_index(self, n_years: int, freq: str) -> pd.DatetimeIndex:
        """
        Computes a DatetimeIndex for forecasting n future years, taking into account frequency mismatches.
        
        Args:
            n_years (int): the number of years to forecast.
            freq (str): the initial frequency: "MS", "Q" or "Y".

        Returns:
            pd.DatetimeIndex:years the appropriate DatetimeIndex for the time series.
        """
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("The index should be a DatetimeIndex.")
        
        last_date = self.data.index[-1]
        
        if freq == "MS":
            start = pd.Timestamp(f"{last_date.year + 1}-01-01")
            end = pd.Timestamp(f"{last_date.year + n_years}-12-01")
        elif freq == "QE":
            start = pd.Timestamp(f"{last_date.year + 1}-Q1")
            end = pd.Timestamp(f"{last_date.year + n_years}-Q4")
        elif freq == "Y":
            start = pd.Timestamp(f"{last_date.year + 1}-12-31")
            end = pd.Timestamp(f"{last_date.year + n_years}-12-31")
        else:
            raise ValueError(f"Unsupported frequency: {freq}")
        
        return pd.date_range(start = start, end = end, freq = freq)
    
    def to_numpy(
        self,
        use_predicted: bool = True,
        allow_missing: bool = False
    ) -> np.ndarray:
        source = self.predicted if use_predicted and self.predicted is not None else self.data
        if not isinstance(source, (pd.Series, pd.DataFrame)):
            return TypeError(f"Expected a pd.Series or pd.DataFrame, got {type(source)}.")
        
        if not allow_missing and source.isna().any():
            raise ValueError("NaN values detected. Set allow_missing=True to override.")
        
        if isinstance(source, pd.Series):
            return source.values.reshape(-1, 1).astype(np.float32)
        
        if isinstance(source, pd.DataFrame):
            return source.data.to_numpy().astype(np.float32)
    
    def to_tensor(
            self,
            use_predicted: bool = True,
            target_col: str = "bankrupt_2024",
            add_batch_dim: bool = False,
            allow_missing: bool = False
        ) -> torch.Tensor:
        """
        Exports the time series to a PyTorch tensor.

        Args:
            use_predicted (bool, optional): toggles using predicted (T) or historical (F) data. Defaults to True.
            target_col (str, optional): identifier of the target column. Defaults to "yhat".
            add_batch_dim (bool, optional): toggles using batch dim or not. Defaults to False.
            allow_missing (bool, optional): if True, does not raise an exception if NaN values are present. Defaults to False.

        Raises:
            ValueError: If target_col is not found in data
            TypeError: If the data type is not a Pandas Series or DataFrame
            ValueError: If NaN values are present and allow_missing is not True

        Returns:
            torch.Tensor: Tensor containing the time series
        """
        source = self.predicted if use_predicted and self.predicted is not None else self.data
        labels = None
        
        if isinstance(source, pd.Series):
            values = source.values
        elif isinstance(source, pd.DataFrame):
            values = source.drop(target_col, axis = 1).values
            if target_col not in source.columns:
                raise ValueError(f"Column '{target_col} not found in DataFrame.")
            labels = source[target_col].values.reshape(-1, 1)
        else:
            raise TypeError("Expected Series or DataFrame.")

        if not allow_missing and pd.isna(values).any():
            raise ValueError("NaN values detected. Set allow_missing=True to override.")
            
        values_tensor = torch.tensor(values, dtype=torch.float32)
        if add_batch_dim:
            values = values_tensor.unsqueeze(0)
        
        if labels is None:
            return values_tensor    
        else:
            labels_tensor = torch.tensor(labels, dtype=torch.float32)
            return values_tensor, labels_tensor 

class MacroTimeSeries(TimeSeries):
    def __init__(
        self,
        data: pd.Series,
        years: list[int],
        n_periods: int = 0,
        order: tuple = (1, 1, 1),
        method: str = "prophet"
    ):
        super().__init__(data)
        self.past_data = None
        self.future_data = None
        self.years=years
        
        if years:
            self._cut()
    
    def _cut(self) -> pd.Series:
        first_year, last_year = self.years[0], self.years[-1]
        
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise TypeError("TimeSeries index must be a pd.DatetimeIndex")
        
        # ---- 1. Select past data from first_year to last_year ----
        self.past_data = self.data.loc[
            datetime.date(year=first_year, month=1, day=1):
            datetime.date(year=last_year, month=12, day=31)
        ]
        
        # ---- 2. Select future data for the next year ----
        future_start = datetime.date(year=last_year+1, month=1, day=1)
        
        self.future_data = self.data.loc[future_start:]
        
        # ---- 3. If future data is incomplete, forecast ----
        expected_end = datetime.date(year=last_year+1, month=12, day=31)
        if self.future_data is None or self.future_data.index[-1].date() < expected_end:
            forecast = self._forecast(method="prophet", n_periods=12)
            
            self.future_data = pd.concat([self.future_data, forecast])
        
        self.data = pd.concat([self.past_data, self.future_data])
        return self.data
    
    def _forecast(self, method: str, n_periods: int, order: Optional[tuple]=None):
        if method == "prophet":
            forecast = self.forecast_prophet(n_periods)
        elif method == "arima":
            forecast = self.forecast_ARIMA(n_periods, order)
        else:
            raise ValueError("Either Prophet or ARIMA is used to produce forecasts")
        
        return forecast
    
    def forecast_ARIMA(
        self,    
        n_periods: int,
        order: tuple
    ) -> pd.DataFrame:
        """
        Forecasts a univariate time series following an ARIMA model.
        This function is used to forecast the _macroeconomic data_.

        Args:
            series (_type_): input time series data
            n_periods (int, optional): n of periods to forecast into the future. Defaults to 3.
            order (tuple, optional): order of the ARIMA model. Defaults to (1, 1, 1).
        """
        
        model = ARIMA(self.data, order = order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps = n_periods)
        
        last_date = self.data.index[-1]
        freq = pd.infer_freq(self.data.index)
        if freq is None:
            freq = "Y" # Assume annual by default
        
        forecast.index = pd.date_range(
            start=last_date + pd.DateOffset(years=1), periods = n_periods, freq = freq
        )
        forecast_df = pd.DataFrame({"yhat": forecast})
        forecast_df.index.name = "year"
        
        return forecast_df

    def forecast_prophet(
        self,
        n_periods: int
    ) -> pd.Series:
        
        data = self.data.copy()
        data = pd.DataFrame([data.index, data.values], index = ["ds", "y"]).T
        
        model = Prophet()
        model_fit = model.fit(data)
        
        freq = pd.infer_freq(data["ds"])
        if freq is None:
            freq = "Y"
        
        future = model_fit.make_future_dataframe(periods = n_periods, freq = freq)
        forecast = model_fit.predict(future)
        
        forecast = forecast.set_index("ds")["yhat"]
        return forecast[-n_periods:]
    
    def average_per_year(self) -> pd.Series:
        return self.forecast.resample("YE").mean()

class FinancialTimeSeries(TimeSeries):
    def __init__(
        self,
        data: pd.DataFrame
    ):
        super().__init__(data)
        
        self.predicted = self.data
        # self.predicted = self.forecast(n_periods)
    
    def forecast(
        self
    ) -> pd.DataFrame:
        """
        Forecasts a univariate time series using Meta's Prophet model.
        This function is used to forecast _company-level financial data_.

        Args:
            series (_type_): input time series data
            n_periods (int, optional): n of periods to forecast into the future. Defaults to 3.
        """
        model = Prophet()
        model_fit = model.fit(self.data)
        
        freq = pd.infer_freq(self.data.index)
        if freq is None:
            freq = "Y" # Assume annual by default
        future = model_fit.make_future_dataframe(
            periods = self.n_periods, freq = freq
        )
        
        forecast = model_fit.predict(future)
        
        return forecast[["ds", "yhat"]]
