"""Train the Logistic Regression baseline, with and without class weights.

- 5-fold stratified CV on the training split (mean and std across folds)
- then fit on the full training split and evaluate on the validation split
- the test split is NOT used here; it is kept for the final comparison

Usage:
    python scripts/train_baseline.py
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, data, evaluate, features, models  # noqa: E402

MODELS_DIR = config.ROOT / "models"

VARIANTS = {
    "LR": None,
    "LR (class weights)": "balanced",
}


def main() -> None:
    train, val, _ = data.load_splits()
    X_train, y_train = features.split_xy(train)
    X_val, y_val = features.split_xy(val)
    columns = list(X_train.columns)

    cv_rows, val_rows, val_scores = {}, {}, {}
    for name, class_weight in VARIANTS.items():
        print(f"\n== {name} ==")
        pipe = models.make_logreg(columns, class_weight=class_weight)

        cv = models.cross_validate_model(pipe, X_train, y_train)
        cv_rows[name] = cv
        print(f"CV PR-AUC {cv['cv_pr_auc_mean']:.4f} +/- {cv['cv_pr_auc_std']:.4f} | "
              f"recall {cv['cv_recall_mean']:.4f} | precision {cv['cv_precision_mean']:.4f}")

        pipe.fit(X_train, y_train)
        scores = pipe.predict_proba(X_val)[:, 1]
        val_scores[name] = scores
        val_rows[name] = evaluate.metrics_at_threshold(y_val, scores, threshold=0.5)

        MODELS_DIR.mkdir(exist_ok=True)
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(pipe, MODELS_DIR / f"{slug}.joblib")

    cv_table = pd.DataFrame(cv_rows).T.round(4)
    val_table = evaluate.results_table(val_rows)
    evaluate.save_table(cv_table, "baseline_cv")
    evaluate.save_table(val_table, "baseline_val")

    print("\nCross-validation on training split (threshold 0.5):")
    print(cv_table.to_string())
    print("\nValidation split (threshold 0.5):")
    print(val_table.to_string())

    evaluate.plot_pr_curves(y_val, val_scores, "Logistic Regression baseline (validation)",
                            name="baseline_pr_curves")
    for name, scores in val_scores.items():
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        evaluate.plot_confusion(y_val, scores, 0.5, name, name=f"cm_{slug}")


if __name__ == "__main__":
    main()
