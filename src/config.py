"""Project-wide settings: paths, random seed and split sizes."""

from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

RAW_CSV = DATA_RAW / "creditcard.csv"
KAGGLE_DATASET = "mlg-ulb/creditcardfraud"

# Reproducibility
RANDOM_STATE = 42

# Stratified split: 60% train, 20% validation, 20% test
TEST_SIZE = 0.20
VAL_SIZE = 0.20  # share of the full dataset

# Cross-validation
CV_FOLDS = 5

TARGET = "Class"
