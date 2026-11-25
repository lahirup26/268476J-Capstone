"""Scoring demo: score transactions and return a fraud probability and a decision.

The transactions come from the held-out test split, so they already carry the
engineered features (the rolling features need the transactions before them,
so a single raw transaction cannot be scored on its own).

Usage:
    python scripts/score.py --sample fraud --n 3
    python scripts/score.py --sample normal --n 3
    python scripts/score.py --index 120
    python scripts/score.py --model lr --threshold 0.145 --sample fraud --n 5
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, data, features  # noqa: E402

MODELS = {"rf": "rf_smote.joblib", "lr": "lr.joblib"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score transactions from the test split.")
    p.add_argument("--model", choices=MODELS, default="rf", help="rf = Random Forest (SMOTE), lr = Logistic Regression")
    p.add_argument("--threshold", type=float, default=0.5, help="decision threshold (default 0.5)")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--index", type=int, help="row number in the test split")
    group.add_argument("--sample", choices=["fraud", "normal", "any"], default="any",
                       help="pick random transactions of this class")
    p.add_argument("--n", type=int, default=5, help="how many transactions to sample")
    p.add_argument("--seed", type=int, default=None, help="random seed for sampling")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_path = config.ROOT / "models" / MODELS[args.model]
    if not model_path.exists():
        sys.exit(f"{model_path} not found. Run the training scripts first.")
    model = joblib.load(model_path)

    _, _, test = data.load_splits()
    if args.index is not None:
        rows = test.iloc[[args.index]]
    else:
        pool = test
        if args.sample == "fraud":
            pool = test[test[config.TARGET] == 1]
        elif args.sample == "normal":
            pool = test[test[config.TARGET] == 0]
        rows = pool.sample(n=min(args.n, len(pool)), random_state=args.seed)

    X, y = features.split_xy(rows)
    proba = model.predict_proba(X)[:, 1]

    out = pd.DataFrame({
        "row": rows.index,
        "hour": rows["hour"].to_numpy(),
        "amount": rows["Amount"].round(2).to_numpy(),
        "fraud_probability": proba.round(4),
        "decision": ["FRAUD - review" if p >= args.threshold else "ok" for p in proba],
        "actual": ["fraud" if v == 1 else "normal" for v in y],
    })
    out["correct"] = ((proba >= args.threshold) == (y.to_numpy() == 1))

    print(f"Model: {MODELS[args.model]} | threshold: {args.threshold}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
