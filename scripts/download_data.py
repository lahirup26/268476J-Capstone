"""Download the ULB credit card fraud dataset and check it.

Usage:
    python scripts/download_data.py

If you are logged in to Kaggle (run `kaggle auth login` once, or set the
KAGGLE_API_TOKEN environment variable), the dataset is downloaded automatically. Otherwise, download creditcard.csv by hand from
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place it in
data/raw/. The script then only checks the file.
"""

import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

EXPECTED_ROWS = 284_807
EXPECTED_FRAUD = 492
EXPECTED_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]


def download() -> None:
    """Download and unzip the dataset with the Kaggle API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("The kaggle package is not installed. Run: pip install -r requirements.txt")

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as err:  # missing or invalid token
        sys.exit(
            f"Kaggle authentication failed ({err}).\n"
            "Either run `kaggle auth login`, or download creditcard.csv by hand from\n"
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud into data/raw/."
        )

    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {config.KAGGLE_DATASET} ...")
    api.dataset_download_files(config.KAGGLE_DATASET, path=config.DATA_RAW, quiet=False)

    for zip_path in config.DATA_RAW.glob("*.zip"):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(config.DATA_RAW)
        zip_path.unlink()


def check(path: Path) -> None:
    """Check that the file has the expected shape, columns and fraud count."""
    df = pd.read_csv(path)
    problems = []
    if list(df.columns) != EXPECTED_COLUMNS:
        problems.append("columns do not match the expected layout")
    if len(df) != EXPECTED_ROWS:
        problems.append(f"expected {EXPECTED_ROWS:,} rows, found {len(df):,}")
    n_fraud = int(df["Class"].sum()) if "Class" in df else -1
    if n_fraud != EXPECTED_FRAUD:
        problems.append(f"expected {EXPECTED_FRAUD} fraud rows, found {n_fraud}")

    if problems:
        sys.exit("Dataset check failed: " + "; ".join(problems))

    print(f"OK: {len(df):,} rows, {n_fraud} fraud ({n_fraud / len(df):.3%}), "
          f"{df.shape[1]} columns.")


def main() -> None:
    if not config.RAW_CSV.exists():
        download()
    if not config.RAW_CSV.exists():
        sys.exit(f"{config.RAW_CSV} not found after download.")
    check(config.RAW_CSV)


if __name__ == "__main__":
    main()
