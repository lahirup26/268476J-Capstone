"""Draw the system architecture diagram (docs/architecture.png).

Usage:
    python scripts/make_architecture_diagram.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

OUT = config.ROOT / "docs" / "architecture.png"

COLORS = {
    "data": "#dbe9f6",
    "prep": "#e3f1dc",
    "model": "#fde9cf",
    "eval": "#efe0f5",
    "out": "#eeeeee",
}


def box(ax, x, y, w, h, title, lines, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=COLORS[kind], ec="#555555", lw=1.2))
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=10.5, weight="bold")
    ax.text(x + w / 2, y + h - 0.68, "\n".join(lines), ha="center", va="top", fontsize=8.3,
            linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, label=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 color="#444444", lw=1.3))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.15, label, ha="center", fontsize=7.5,
                color="#444444", style="italic")


def main() -> None:
    fig, ax = plt.subplots(figsize=(15, 8.6))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.6)
    ax.axis("off")
    ax.set_title("Credit Card Fraud Detection - System Architecture", fontsize=14, weight="bold")

    # Row 1: data pipeline
    box(ax, 0.2, 5.6, 2.8, 2.4, "1. Data source",
        ["Kaggle: mlg-ulb/creditcardfraud", "284,807 transactions", "492 fraud (0.172%)",
         "download_data.py", "(download + validate)"], "data")
    box(ax, 3.6, 5.6, 3.4, 2.4, "2. Cleaning",
        ["prepare_data.py / src/data.py", "remove 1,081 duplicates", "check missing values (none)"],
        "prep")
    box(ax, 7.6, 5.6, 3.4, 2.4, "3. Feature engineering",
        ["src/features.py", "hour of day (+ sin/cos)", "rolling counts and mean Amount",
         "over 1 min / 10 min / 1 h", "PAST rows only (no leakage)"], "prep")
    box(ax, 11.6, 5.6, 3.2, 2.4, "4. Stratified split",
        ["seed 42", "train 60% (284 fraud)", "validation 20% (94 fraud)",
         "test 20% (95 fraud)", "-> data/processed/"], "data")

    arrow(ax, 3.0, 6.8, 3.6, 6.8)
    arrow(ax, 7.0, 6.8, 7.6, 6.8)
    arrow(ax, 11.0, 6.8, 11.6, 6.8)

    # Row 2: model pipeline
    ax.add_patch(FancyBboxPatch((0.2, 2.55), 9.0, 2.45, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc="white", ec="#999999", lw=1, ls="--"))
    ax.text(0.4, 4.82, "5. Model pipeline (imblearn Pipeline, fitted on training data / CV training folds only)",
            fontsize=9, weight="bold", va="top")
    box(ax, 0.5, 2.75, 2.5, 1.75, "RobustScaler",
        ["Time, Amount and", "engineered features", "(V1-V28 unchanged)"], "model")
    box(ax, 3.4, 2.75, 2.4, 1.75, "Imbalance handling",
        ["none / class weights /", "SMOTE (10% ratio)", "train folds only"], "model")
    box(ax, 6.2, 2.75, 2.8, 1.75, "Models",
        ["Baseline: Logistic Regression", "Improved: Random Forest",
         "Comparison: Isolation Forest"], "model")
    arrow(ax, 3.0, 3.6, 3.4, 3.6)
    arrow(ax, 5.8, 3.6, 6.2, 3.6)

    box(ax, 9.8, 2.55, 5.0, 2.45, "6. Evaluation (src/evaluate.py)",
        ["5-fold stratified CV on train (mean +/- std)",
         "validation: PR-AUC, precision, recall, F1",
         "threshold tuning on validation only",
         "test set used once for the final comparison",
         "(M3: McNemar test, cost matrix, error analysis)"], "eval")
    arrow(ax, 13.2, 5.6, 13.2, 5.0, "train / val / test")
    arrow(ax, 9.0, 3.6, 9.8, 3.6)
    arrow(ax, 12.0, 5.6, 4.7, 5.0, "training split")

    # Row 3: outputs
    box(ax, 0.2, 0.2, 4.4, 1.8, "Saved models",
        ["models/*.joblib", "(full pipeline: scaler + model)"], "out")
    box(ax, 5.2, 0.2, 4.4, 1.8, "Scoring demo",
        ["score.py: one transaction ->", "fraud probability + decision"], "out")
    box(ax, 10.2, 0.2, 4.6, 1.8, "Results",
        ["results/tables/*.csv", "results/figures/*.png", "(PR curves, confusion matrices)"], "out")
    arrow(ax, 7.6, 2.75, 2.4, 2.0)
    arrow(ax, 2.4 + 2.2, 1.1, 5.2, 1.1)
    arrow(ax, 12.3, 2.55, 12.5, 2.0)

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
