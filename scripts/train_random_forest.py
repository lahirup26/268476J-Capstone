"""Train the improved model: Random Forest with different ways to handle the imbalance.

Variants (to test H2 - does imbalance handling help?):
- RF                  no imbalance handling
- RF (class weights)  class_weight="balanced_subsample"
- RF (SMOTE)          SMOTE inside the pipeline (training folds only)

- 5-fold stratified CV on the training split (mean and std across folds)
- then fit on the full training split and evaluate on the validation split
- the test split is NOT used here

Usage:
    python scripts/train_random_forest.py
"""

import sys
import time
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, data, evaluate, features, models  # noqa: E402

MODELS_DIR = config.ROOT / "models"

VARIANTS = {
    "RF": dict(class_weight=None, smote=False),
    "RF (class weights)": dict(class_weight="balanced_subsample", smote=False),
    "RF (SMOTE)": dict(class_weight=None, smote=True),
}


def slugify(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


def main() -> None:
    train, val, _ = data.load_splits()
    X_train, y_train = features.split_xy(train)
    X_val, y_val = features.split_xy(val)
    columns = list(X_train.columns)

    cv_rows, val_rows, val_scores = {}, {}, {}
    for name, params in VARIANTS.items():
        print(f"\n== {name} ==", flush=True)
        start = time.time()
        pipe = models.make_random_forest(columns, **params)

        # the forest already uses all CPU cores, so the folds run one by one
        cv = models.cross_validate_model(pipe, X_train, y_train, n_jobs=1)
        cv_rows[name] = cv
        print(f"CV PR-AUC {cv['cv_pr_auc_mean']:.4f} +/- {cv['cv_pr_auc_std']:.4f} | "
              f"recall {cv['cv_recall_mean']:.4f} | precision {cv['cv_precision_mean']:.4f}",
              flush=True)

        pipe.fit(X_train, y_train)
        scores = pipe.predict_proba(X_val)[:, 1]
        val_scores[name] = scores
        val_rows[name] = evaluate.metrics_at_threshold(y_val, scores, threshold=0.5)

        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(pipe, MODELS_DIR / f"{slugify(name)}.joblib")
        print(f"done in {time.time() - start:.0f}s", flush=True)

    cv_table = pd.DataFrame(cv_rows).T.round(4)
    val_table = evaluate.results_table(val_rows)
    evaluate.save_table(cv_table, "random_forest_cv")
    evaluate.save_table(val_table, "random_forest_val")

    print("\nCross-validation on training split (threshold 0.5):")
    print(cv_table.to_string())
    print("\nValidation split (threshold 0.5):")
    print(val_table.to_string())

    evaluate.plot_pr_curves(y_val, val_scores, "Random Forest variants (validation)",
                            name="random_forest_pr_curves")
    for name, scores in val_scores.items():
        evaluate.plot_confusion(y_val, scores, 0.5, name, name=f"cm_{slugify(name)}")


if __name__ == "__main__":
    main()
