"""Model definitions. Every model is a pipeline, so preprocessing (scaling)
is fitted only on the data the model is trained on."""

from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from src import config
from src.features import make_preprocessor


def make_logreg(columns: list[str], class_weight: str | None = None) -> Pipeline:
    """Logistic Regression baseline: RobustScaler (on non-PCA columns) + LR."""
    return Pipeline(
        [
            ("prep", make_preprocessor(columns)),
            ("model", LogisticRegression(
                class_weight=class_weight,
                max_iter=2000,
                random_state=config.RANDOM_STATE,
            )),
        ]
    )


def cross_validate_model(pipeline, X, y) -> dict:
    """Stratified k-fold CV on the training data.

    Returns the mean and standard deviation of PR-AUC, precision, recall and F1
    (at the default 0.5 threshold) across the folds.
    """
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    scores = cross_validate(
        pipeline, X, y, cv=cv, n_jobs=-1,
        scoring={
            "pr_auc": "average_precision",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
        },
    )
    out = {}
    for name in ("pr_auc", "precision", "recall", "f1"):
        values = scores[f"test_{name}"]
        out[f"cv_{name}_mean"] = float(values.mean())
        out[f"cv_{name}_std"] = float(values.std())
    return out
