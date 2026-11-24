"""Feature engineering and preprocessing.

Row-wise features (like hour of day) are added before the split, because they
only use the row itself. Scaling is NOT done here: the scaler is part of the
model pipeline, so it is fitted on the training data (or training folds) only.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

from src import config

PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
TIME_FEATURES = ["hour", "hour_sin", "hour_cos"]

# Columns with large or very different ranges that need scaling.
# RobustScaler uses the median and IQR, so the big outliers in Amount
# do not dominate the scaling.
SCALED_BASE = ["Time", "Amount"]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour of day from Time (seconds since the first transaction).

    hour      0-23, hour 0 is the start of the recording (not real midnight)
    hour_sin  \\ cyclic encoding, so hour 23 and hour 0 are close together
    hour_cos  /
    """
    df = df.copy()
    df["hour"] = (df["Time"] // 3600 % 24).astype(int)
    angle = 2 * np.pi * df["hour"] / 24
    df["hour_sin"] = np.sin(angle)
    df["hour_cos"] = np.cos(angle)
    return df


ROLLING_WINDOWS = [60, 600, 3600]  # seconds: 1 minute, 10 minutes, 1 hour

ROLLING_FEATURES = (
    [f"tx_count_{w}s" for w in ROLLING_WINDOWS]
    + [f"amount_mean_{w}s" for w in ROLLING_WINDOWS[1:]]
    + ["amount_ratio_3600s", "secs_since_prev"]
)


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling count and amount features that use PAST transactions only.

    The dataset has no card or customer ID, so the windows are over the whole
    stream of transactions. They describe the recent activity around each
    transaction (how busy the system is, the usual amount right now).

    Leakage control:
    - rows are sorted by Time, and every window is [t - w, t), so it ends
      strictly before the current transaction (closed="left"). Rows with the
      same Time are not counted either.
    - the labels (Class) are never used.

    tx_count_{w}s       number of transactions in the previous w seconds
    amount_mean_{w}s    mean Amount in the previous w seconds
    amount_ratio_3600s  Amount / (mean Amount in the previous hour + 1),
                        how unusual this amount is compared with recent ones
    secs_since_prev     seconds since the previous transaction (0 for the first one)
    """
    df = df.copy()
    order = df["Time"].sort_values(kind="stable").index
    s = df.loc[order]
    amount = pd.Series(s["Amount"].to_numpy(), index=pd.to_timedelta(s["Time"], unit="s"))

    new = pd.DataFrame(index=order)
    for w in ROLLING_WINDOWS:
        roll = amount.rolling(f"{w}s", closed="left")
        new[f"tx_count_{w}s"] = roll.count().fillna(0).to_numpy()
        if w != ROLLING_WINDOWS[0]:
            # no past rows in the window -> use the current amount (ratio = about 1)
            new[f"amount_mean_{w}s"] = roll.mean().fillna(amount).to_numpy()

    new["amount_ratio_3600s"] = s["Amount"].to_numpy() / (new["amount_mean_3600s"] + 1)
    new["secs_since_prev"] = s["Time"].diff().fillna(0).to_numpy()

    return df.join(new)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All model input columns (everything except the target)."""
    return [c for c in df.columns if c != config.TARGET]


def make_preprocessor(columns: list[str]) -> ColumnTransformer:
    """RobustScaler on Time, Amount and engineered numeric features; V1-V28 pass through.

    V1-V28 are already PCA outputs with mean about 0, so they are left as they are.
    """
    scaled = [c for c in columns if c not in PCA_FEATURES and c not in ("hour_sin", "hour_cos")]
    passthrough = [c for c in columns if c not in scaled]
    return ColumnTransformer(
        [
            ("robust", RobustScaler(), scaled),
            ("keep", "passthrough", passthrough),
        ],
        verbose_feature_names_out=False,
    )


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into features X and target y."""
    return df[feature_columns(df)], df[config.TARGET]
