"""Load the raw data, remove duplicates and make a stratified train/val/test split."""

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def load_raw() -> pd.DataFrame:
    """Read creditcard.csv from data/raw."""
    if not config.RAW_CSV.exists():
        raise FileNotFoundError(
            f"{config.RAW_CSV} not found. Run: python scripts/download_data.py"
        )
    return pd.read_csv(config.RAW_CSV)


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop exact duplicate rows (keep the first one).

    Duplicates are removed before splitting so that the same transaction
    cannot appear in both the training and the test set.
    """
    dup_mask = df.duplicated(keep="first")
    info = {
        "rows_before": len(df),
        "duplicates": int(dup_mask.sum()),
        "fraud_duplicates": int(df.loc[dup_mask, config.TARGET].sum()),
    }
    df = df.loc[~dup_mask].reset_index(drop=True)
    info["rows_after"] = len(df)
    return df, info


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 60/20/20 train/validation/test split with a fixed seed."""
    train_val, test = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        stratify=df[config.TARGET],
        random_state=config.RANDOM_STATE,
    )
    val_share = config.VAL_SIZE / (1 - config.TEST_SIZE)
    train, val = train_test_split(
        train_val,
        test_size=val_share,
        stratify=train_val[config.TARGET],
        random_state=config.RANDOM_STATE,
    )
    return train, val, test


def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """Save the three splits to data/processed (not tracked by git)."""
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(config.DATA_PROCESSED / f"{name}.csv", index=False)


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the saved splits back from data/processed."""
    return tuple(
        pd.read_csv(config.DATA_PROCESSED / f"{name}.csv")
        for name in ("train", "val", "test")
    )
