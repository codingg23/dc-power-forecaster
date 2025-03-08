"""
preprocessing.py

Data preprocessing pipeline for the power forecaster.

Takes raw sensor data (from DCIM or synthetic generator) and produces
windowed feature/target tensors ready for model training.

Main steps:
  1. Resample to target frequency (raw data might be 1-min, we want 15-min)
  2. Handle missing values (sensors drop out, DCIM systems have gaps)
  3. Build time features (cyclic encoding of hour/dow)
  4. Normalise
  5. Sliding window into (context, horizon) pairs

One thing I got wrong early on: normalised across the entire dataset before
splitting train/val/test, which leaks future stats into training. Fixed to
fit scaler on train set only.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field
from typing import Optional, Tuple
import pickle
import logging

logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    target_col: str = "pdu_kw"
    freq: str = "15min"
    context_len: int = 672       # 7 days @ 15min
    horizon: int = 96            # 24h @ 15min
    max_gap_fill: int = 4        # max consecutive NaN to interpolate (steps)
    train_frac: float = 0.7
    val_frac: float = 0.15
    # test_frac is implied = 1 - train - val


def cyclic_encode(series: pd.Series, period: float) -> Tuple[pd.Series, pd.Series]:
    """Encode a periodic feature as (sin, cos) to avoid discontinuity at boundaries."""
    angle = 2 * np.pi * series / period
    return np.sin(angle), np.cos(angle)


def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclic time features to a DataFrame with a DatetimeIndex.
    These are almost always the most important features for DC load.
    """
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")

    hour = idx.hour + idx.minute / 60.0
    dow = idx.dayofweek.astype(float)

    df["hour_sin"], df["hour_cos"] = cyclic_encode(pd.Series(hour, index=idx), 24.0)
    df["dow_sin"], df["dow_cos"] = cyclic_encode(pd.Series(dow, index=idx), 7.0)
    df["is_weekend"] = (idx.dayofweek >= 5).astype(float)

    return df


class PowerDataset:
    """
    Sliding-window dataset for power forecasting.

    Handles:
    - Resampling (aggregation method matters: sum for energy, mean for load)
    - Gap filling
    - Feature engineering
    - Train/val/test split (time-based, not random)
    - Normalisation (fit on train only)
    """

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.scaler = StandardScaler()
        self._fitted = False

    def load_and_preprocess(self, path: str) -> pd.DataFrame:
        """Load raw power data and preprocess into model-ready format."""
        logger.info(f"Loading data from {path}")

        if path.endswith(".parquet"):
            raw = pd.read_parquet(path)
        else:
            raw = pd.read_csv(path, parse_dates=["timestamp"])

        # aggregate to facility level — sum across all racks
        # this assumes the raw data is per-rack
        if "rack_id" in raw.columns:
            logger.info("Aggregating per-rack data to facility level")
            raw = (
                raw.groupby("timestamp")["pdu_kw"]
                .sum()
                .reset_index()
                .rename(columns={"pdu_kw": "facility_kw"})
            )
            # update target col if needed
            if self.config.target_col == "pdu_kw":
                self.config.target_col = "facility_kw"

        df = raw.set_index("timestamp")
        df.index = pd.to_datetime(df.index)

        # resample to target frequency
        target_col = self.config.target_col
        df = df[[target_col]].resample(self.config.freq).mean()

        # gap handling
        n_gaps = df[target_col].isna().sum()
        if n_gaps > 0:
            logger.warning(f"Found {n_gaps} missing values, interpolating gaps ≤ {self.config.max_gap_fill} steps")
            df[target_col] = df[target_col].interpolate(
                method="linear",
                limit=self.config.max_gap_fill,
                limit_direction="forward",
            )
            remaining_nans = df[target_col].isna().sum()
            if remaining_nans > 0:
                logger.warning(f"{remaining_nans} values still missing after interpolation — dropping")
                df = df.dropna()

        # first difference as an additional feature (rate of change)
        df["delta_kw"] = df[target_col].diff().fillna(0)

        # time features
        df = build_time_features(df)

        logger.info(f"Preprocessed data: {len(df)} steps, {df.index[0]} to {df.index[-1]}")
        return df

    def train_val_test_split(self, df: pd.DataFrame):
        n = len(df)
        train_end = int(n * self.config.train_frac)
        val_end = int(n * (self.config.train_frac + self.config.val_frac))

        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]

        logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
        return train, val, test

    def fit_scaler(self, train_df: pd.DataFrame):
        """Fit normalisation on training data only."""
        self.scaler.fit(train_df.values)
        self._fitted = True

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit_scaler before transform")
        return self.scaler.transform(df.values)

    def inverse_transform_target(self, predictions: np.ndarray) -> np.ndarray:
        """Inverse transform just the target column."""
        target_idx = 0  # first column after preprocessing
        dummy = np.zeros((len(predictions), self.scaler.n_features_in_))
        dummy[:, target_idx] = predictions
        return self.scaler.inverse_transform(dummy)[:, target_idx]

    def make_windows(self, arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows from normalised array.

        Returns:
            X: (n_windows, context_len, n_features)
            y: (n_windows, horizon)  — target feature only (index 0)
        """
        ctx = self.config.context_len
        hor = self.config.horizon
        n = len(arr)

        X, y = [], []
        for i in range(n - ctx - hor + 1):
            X.append(arr[i:i + ctx])
            y.append(arr[i + ctx:i + ctx + hor, 0])  # target is first col

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def save_scaler(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Scaler saved to {path}")

    @classmethod
    def load_scaler(cls, path: str, config: Optional[PreprocessConfig] = None):
        instance = cls(config or PreprocessConfig())
        with open(path, "rb") as f:
            instance.scaler = pickle.load(f)
        instance._fitted = True
        return instance
