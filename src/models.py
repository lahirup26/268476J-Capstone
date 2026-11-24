"""Model definitions. Every model is a pipeline, so preprocessing (scaling)
is fitted only on the data the model is trained on."""

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
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


# SMOTE oversamples fraud up to 10% of the normal class (not 50/50).
# A full 50/50 balance would add about 170,000 synthetic rows, which is slow
# and creates many synthetic points far from real fraud.
SMOTE_RATIO = 0.1


def make_random_forest(
    columns: list[str],
    class_weight: str | None = None,
    smote: bool = False,
    n_estimators: int = 300,
) -> Pipeline:
    """Random Forest, optionally with class weights and/or SMOTE.

    SMOTE is a step INSIDE the imblearn pipeline, so during cross-validation it
    is fitted only on the training folds; the validation fold keeps its real
    class balance and never contains synthetic rows.
    """
    steps = [("prep", make_preprocessor(columns))]
    if smote:
        steps.append(("smote", SMOTE(sampling_strategy=SMOTE_RATIO,
                                     random_state=config.RANDOM_STATE)))
    steps.append(("model", RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
    )))
    return Pipeline(steps)


def cross_validate_model(pipeline, X, y, n_jobs: int = -1) -> dict:
    """Stratified k-fold CV on the training data.

    Returns the mean and standard deviation of PR-AUC, precision, recall and F1
    (at the default 0.5 threshold) across the folds.
    """
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    scores = cross_validate(
        pipeline, X, y, cv=cv, n_jobs=n_jobs,
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
