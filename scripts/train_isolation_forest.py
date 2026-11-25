"""Isolation Forest: unsupervised anomaly detection (H3).

- fitted on the training split WITHOUT labels
- threshold also chosen without labels: flag the top 0.17% most unusual
  training transactions (the known fraud rate), then apply it to validation
- compared with the supervised models on the validation split

Usage:
    python scripts/train_isolation_forest.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, data, evaluate, features, models  # noqa: E402

MODELS_DIR = config.ROOT / "models"
SUPERVISED = {"LR": "lr.joblib", "RF (SMOTE)": "rf_smote.joblib"}


def main() -> None:
    train, val, _ = data.load_splits()
    X_train, y_train = features.split_xy(train)
    X_val, y_val = features.split_xy(val)

    pipe = models.make_isolation_forest(list(X_train.columns))
    pipe.fit(X_train)  # no labels
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, MODELS_DIR / "isolation_forest.joblib")

    fraud_rate = y_train.mean()
    threshold = float(np.quantile(models.anomaly_scores(pipe, X_train), 1 - fraud_rate))
    if_scores = models.anomaly_scores(pipe, X_val)

    rows = {"Isolation Forest": evaluate.metrics_at_threshold(y_val, if_scores, threshold)}
    scores = {"Isolation Forest": if_scores}

    # supervised models for comparison (trained by the earlier scripts)
    for name, file in SUPERVISED.items():
        path = MODELS_DIR / file
        if not path.exists():
            print(f"skip {name}: {path} missing (run the training scripts first)")
            continue
        model = joblib.load(path)
        s = model.predict_proba(X_val)[:, 1]
        scores[name] = s
        rows[name] = evaluate.metrics_at_threshold(y_val, s, 0.5)

    table = evaluate.results_table(rows)
    evaluate.save_table(table, "isolation_forest_val")
    print(f"Isolation Forest threshold (top {fraud_rate:.3%} of training scores): {threshold:.4f}")
    print("\nValidation split:")
    print(table.to_string())

    evaluate.plot_pr_curves(y_val, scores, "Unsupervised vs supervised (validation)",
                            name="isolation_forest_pr_curves")
    evaluate.plot_confusion(y_val, if_scores, threshold, "Isolation Forest",
                            name="cm_isolation_forest")


if __name__ == "__main__":
    main()
