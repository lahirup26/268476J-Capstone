"""Remove duplicates, split the data and save the splits to data/processed.

Usage:
    python scripts/prepare_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, data  # noqa: E402


def main() -> None:
    df = data.load_raw()
    df, info = data.remove_duplicates(df)
    print(f"Rows: {info['rows_before']:,} -> {info['rows_after']:,} "
          f"(removed {info['duplicates']:,} duplicates, {info['fraud_duplicates']} of them fraud)")

    train, val, test = data.split(df)
    data.save_splits(train, val, test)

    print(f"{'split':<6}{'rows':>10}{'fraud':>8}{'fraud %':>10}")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        n_fraud = int(part[config.TARGET].sum())
        print(f"{name:<6}{len(part):>10,}{n_fraud:>8}{n_fraud / len(part):>10.3%}")
    print(f"Saved to {config.DATA_PROCESSED}")


if __name__ == "__main__":
    main()
