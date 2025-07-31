import pandas as pd
import numpy as np
from pathlib import Path
import logging

from src.data.insee import get_insee_series

logger = logging.getLogger(__name__)

class CompanyDataLoader:
    def __init__(self, path: Path, sheet_name: str="Results", revenue_cap=3_000):
        if isinstance(path, tuple):
            if len(path) != 1:
                raise ValueError(f"Expected a single path, got tuple of length {len(path)}: {path}")
            path = path[0]
        
        self.path=path
        self.sheet_name=sheet_name
        self.revenue_cap=revenue_cap
        self.company_col=None
        self.bankruptcy_col=None
        self.years=None
        self.mode=None
        
    def load(self, company_col: str, bankruptcy_col: str, mode: str = "train"):
        self.company_col=company_col
        self.bankruptcy_col=bankruptcy_col
        self.mode = mode
        
        df = self._read_file()
        if self.company_col not in df.columns:
            raise ValueError(f"Expected company_col '{self.company_col} in columns")
        
        self.years = sorted(
            {int(col[-4:]) for col in df.columns if "revenue" in col}
        )
        
        return self._clean(df)
        
    def _read_file(self) -> pd.DataFrame:
        logger.info(f"Reading file: {self.path}")
        return pd.read_excel(
            self.path, 
            sheet_name=self.sheet_name, 
            na_values=["n.a."]
        )
        
    def _clean(self, df: pd.DataFrame):
        """_summary_
        Method for cleaning financial data.
        Data must come from BVD's Orbis platform and be exported according to a specific format.
        
        Raises:
            ValueError: absence of the "Turnover USD 2023" column.
            ValueError: mismatch between actual and expected numbers of columns.

        Returns:
            pd.DataFrame: clean dataframe
        """
        # ---- Sanity check ----
        revenue_col = f"Operating revenue (Turnover)\nth USD {self.years[-1]}" 
        if revenue_col not in df.columns:
            raise ValueError("Missing expected revenue column in input data.")
        
        # ---- Filtering companies based on revenue ----
        logger.info("Dropping high-revenue outliers...")
        df = df[df[revenue_col] <= self.revenue_cap].copy()
        
        # ---- Target variable generation ----
        if self.mode == "train" and self.bankruptcy_col:
            max_attempts=10
            for attempt in range(max_attempts):
                try:
                    df[self.bankruptcy_col] = pd.to_datetime(
                        df[self.bankruptcy_col], origin="1899-12-30", unit="D", errors="coerce"
                    )
                    
                    target_year = int(self.years[-1])
                    bankrupt_map = (
                        df.dropna(subset=[self.bankruptcy_col])
                            .assign(year=lambda d: d[self.bankruptcy_col].dt.year)
                            .groupby(self.company_col)["year"]
                            .agg(lambda years: (target_year <= years.values).any())
                    )
                    
                    df[f"bankrupt_{target_year}"] = df[self.company_col].map(bankrupt_map).fillna(0).astype(int)
                    break
                except Exception as e:
                    if attempt == (max_attempts - 1):
                        raise RuntimeError(f"Failed to process data after {max_attempts} attempts")
                    
        
        df.drop(self.bankruptcy_col, axis = 1, inplace=True)
        df.dropna(axis=0, inplace=True)
        
        # ---- Renaming the variables ----
        column_aliases = {
            "Operating revenue (Turnover)\nth USD ": "revenue",
            "P/L before tax\nth USD ": "ebt",
            # "Total assets\nth USD ": "ats",
            "Shareholders funds\nth USD ": "sheq",
            "Cash flow [Net Income before D&A]\nth USD ": "cf",
            # "Current ratio\n": "cr"
        }
        financial_columns = {
            f"{dirty}{year}": f"{clean}_{year}"
            for dirty, clean in column_aliases.items()
            for year in self.years
        }
        
        for old, new in financial_columns.items():
            df[new] = pd.to_numeric(df[old], errors="coerce")
            
        df = df.drop(financial_columns.keys(), errors="ignore", axis=1)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        return df.reset_index(drop=True)

class MacroDataLoader:
    def __init__(self, ids: list[str]):
        self.ids = ids
        
    def load(self) -> pd.DataFrame:
        logger.info(f"Loading {len(self.ids)} macroeconomic series...")
        series = [get_insee_series(id) for id in self.ids]
        
        df = pd.concat(series, axis=1)
        df.columns = self.ids
        
        # Interpolate missing values
        df = df.interpolate(
            method="linear", 
            axis=0,
            limit_direction="both",
            limit=None,
            inplace=None
        )
        
        return df
    
    # OBSOLETE — import function for CSV files
    def _load_series(self, path: Path) -> pd.Series:
        df = pd.read_csv(
            path,
            sep=";",
            skiprows=3,
            usecols=[0,1],
            names=["Date", "Value"],
            header=None
        )
        df["Date"]=pd.to_datetime(df["Date"], errors="coerce")
        df["Value"]=pd.to_numeric(df["Value"], errors="coerce")
        
        df.set_index("Date", inplace=True)
        return df["Value"].sort_index()