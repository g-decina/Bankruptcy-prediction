import sdmx
import pandas as pd
from typing import Optional

def get_insee_series(
    series_id: str,
    start_period: Optional[str]=None,
    end_period: Optional[str]=None
) -> pd.Series:
    c = sdmx.Client("INSEE")
    
    params={k: v for k, v in {
        "startPeriod": start_period,
        "endPeriod": end_period
    }.items() if v is not None}
    
    r = c.get(
        resource_type="data", 
        resource_id=f"SERIES_BDM/{series_id}",
        params=params
    )

    dataset = r.data[0]
    data = []
    
    for obs in dataset.obs:
        time = obs.key.get_values()[1]
        value = obs.value
        
        if value is not None:
            data.append({
                "TIME_PERIOD": time,
                "VALUE": float(value)
            })

    df = pd.DataFrame(data).sort_values("TIME_PERIOD")
    df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"])
    df.set_index("TIME_PERIOD", inplace=True)
    
    series = pd.Series(df["VALUE"]).rename(series_id)
    
    inferred_freq = pd.infer_freq(series.index)
    if inferred_freq:
        series.index.freq = inferred_freq
    
    return series

