"""Choose decision thresholds on the validation split, then evaluate once on the test split.

Two threshold rules, both chosen on VALIDATION data only:
- best F1          threshold with the highest F1
- recall >= 0.80   highest threshold that still reaches recall 0.80
                   (the M1 target), so precision is as high as possible

The test split is used only here, after all choices are fixed, to report the
final numbers. Isolation Forest keeps its label-free threshold.

Usage:
    python scripts/tune_threshold.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, data, evaluate, features, models  # noqa: E402

MODELS_DIR = config.ROOT / "models"
SUPERVISED = {"LR": "lr.joblib", "RF (SMOTE)": "rf_smote.joblib"}
TARGET_RECALL = 0.80


def main() -> None:
    train, val, test = data.load_splits()
    X_train, y_train = features.split_xy(train)
    X_val, y_val = features.split_xy(val)
    X_test, y_test = features.split_xy(test)

    val_rows, test_rows, test_scores = {}, {}, {}

    for name, file in SUPERVISED.items():
        model = joblib.load(MODELS_DIR / file)
        s_val = model.predict_proba(X_val)[:, 1]
        s_test = model.predict_proba(X_test)[:, 1]
        test_scores[name] = s_test

        thresholds = {
            "default 0.5": 0.5,
            "best F1": evaluate.best_f1_threshold(y_val, s_val),
            f"recall>={TARGET_RECALL:.2f}": evaluate.threshold_for_recall(y_val, s_val, TARGET_RECALL),
        }
        for rule, t in thresholds.items():
            val_rows[f"{name} | {rule}"] = evaluate.metrics_at_threshold(y_val, s_val, t)
            test_rows[f"{name} | {rule}"] = evaluate.metrics_at_threshold(y_test, s_test, t)

        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        evaluate.plot_threshold_curve(
            y_val, s_val, f"{name}: threshold vs metrics (validation)",
            marks={k: v for k, v in thresholds.items() if k != "default 0.5"},
            name=f"threshold_curve_{slug}",
        )

    # Isolation Forest: label-free threshold (top fraud-rate share of training scores)
    iso = joblib.load(MODELS_DIR / "isolation_forest.joblib")
    t_iso = float(np.quantile(models.anomaly_scores(iso, X_train), 1 - y_train.mean()))
    s_test_iso = models.anomaly_scores(iso, X_test)
    test_scores["Isolation Forest"] = s_test_iso
    test_rows["Isolation Forest | top 0.17%"] = evaluate.metrics_at_threshold(y_test, s_test_iso, t_iso)

    val_table = evaluate.results_table(val_rows)
    test_table = evaluate.results_table(test_rows)
    evaluate.save_table(val_table, "thresholds_val")
    evaluate.save_table(test_table, "final_test")

    print("Validation (thresholds chosen here):")
    print(val_table.to_string())
    print("\nTest (used once, thresholds fixed from validation):")
    print(test_table.to_string())

    evaluate.plot_pr_curves(y_test, test_scores, "Final comparison (test split)",
                            name="final_test_pr_curves")
    best = test_rows["RF (SMOTE) | best F1"]["threshold"]
    evaluate.plot_confusion(y_test, test_scores["RF (SMOTE)"], best,
                            "RF (SMOTE) on test, best-F1 threshold", name="cm_final_rf_smote")


if __name__ == "__main__":
    main()
