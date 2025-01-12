"""
baseline.py

Naive and Prophet baselines for benchmarking.

Always worth having a simple baseline — it keeps you honest
about how much the fancy model is actually helping.

Naive seasonal: predict next 24h = same 24h last week.
This is surprisingly hard to beat on DC load data because
weekly seasonality is so strong.
"""
import numpy as np
import pandas as pd
from typing import Optional


class NaiveSeasonalBaseline:
    """
    Predicts next H steps = same H steps from 7 days ago.
    No training required.
    """
    def __init__(self, horizon: int = 96, seasonal_period: int = 672):
        self.horizon = horizon
        self.seasonal_period = seasonal_period  # 7 days @ 15min

    def predict(self, history: np.ndarray) -> np.ndarray:
        if len(history) < self.seasonal_period + self.horizon:
            # not enough history — fall back to last known value
            return np.full(self.horizon, history[-1])
        start = len(history) - self.seasonal_period
        return history[start:start + self.horizon]


class ProphetBaseline:
    """
    Facebook Prophet baseline.
    Good for explainability, mediocre accuracy on DC data.
    Keeping it as a reference point.
    """
    def __init__(self, horizon: int = 96, freq: str = "15min"):
        self.horizon = horizon
        self.freq = freq
        self._model = None

    def fit(self, df: pd.DataFrame, target_col: str = "facility_kw"):
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("pip install prophet")

        train = df[[target_col]].copy()
        train.index = pd.to_datetime(train.index)
        prophet_df = train.reset_index().rename(columns={"timestamp": "ds", target_col: "y"})

        self._model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,  # usually not enough data
            changepoint_prior_scale=0.05,
        )
        self._model.fit(prophet_df)
        self._last_ds = prophet_df["ds"].iloc[-1]

    def predict(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() first")
        future = self._model.make_future_dataframe(
            periods=self.horizon, freq=self.freq, include_history=False
        )
        forecast = self._model.predict(future)
        return forecast["yhat"].values
