
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted


# Default classification threshold applied to all confusion-matrix diagnostics
DEFAULT_THRESHOLD = 0.50


# Threshold-free metrics reported alongside the threshold-based diagnostics
PROBABILITY_METRICS = [
    "roc_auc",
    "pr_auc",
    "brier_score",
]

# Metrics that depend on the chosen classification threshold
THRESHOLD_METRICS = [
    "accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
]




def validate_evaluation_inputs(models, X, y):
    """Check that every model can score the evaluation data."""

    # Confirm that predictors and response describe the same observations
    assert len(X) == len(y), (len(X), len(y))

    # Confirm that the response has not leaked into the predictors
    assert "is_canceled" not in X.columns

    # Confirm that both response classes are present
    assert set(pd.unique(y)) == {0, 1}

    for name, model in models.items():

        # Catch result tables or unfitted objects stored by mistake
        if not hasattr(model, "predict_proba"):
            raise TypeError(
                f"`{name}` is not a fitted classifier: {type(model).__name__}"
            )




def predict_probabilities(model, X):
    """Return predicted cancellation probabilities from a fitted pipeline."""

    probabilities = model.predict_proba(X)[:, 1]

    # Confirm that the pipeline produced one usable probability per observation
    assert probabilities.shape == (len(X),)
    assert np.isfinite(probabilities).all()

    return probabilities




def evaluate_model(model, X, y, threshold=DEFAULT_THRESHOLD):
    """Compute threshold-free and threshold-based metrics for one fitted model."""

    probabilities = predict_probabilities(model, X)
    predictions = (probabilities >= threshold).astype(int)

    true_negatives, false_positives, false_negatives, true_positives = (
        confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    )

    return {
        # Threshold-free measures of ranking quality and calibration
        "roc_auc": roc_auc_score(y, probabilities),
        "pr_auc": average_precision_score(y, probabilities),
        "brier_score": brier_score_loss(y, probabilities),

        # Measures that depend on the chosen threshold
        "accuracy": accuracy_score(y, predictions),
        "sensitivity": recall_score(y, predictions, zero_division=0),
        "specificity": true_negatives / (true_negatives + false_positives),
        "precision": precision_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),

        # Raw confusion-matrix counts for reporting
        "true_negatives": int(true_negatives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "true_positives": int(true_positives),
    }




def evaluate_models(models, X, y, threshold=DEFAULT_THRESHOLD):
    """Evaluate every fitted model on the same data and threshold."""

    validate_evaluation_inputs(models, X, y)

    rows = []

    for name, model in models.items():
        result = evaluate_model(model, X, y, threshold=threshold)
        result["model"] = name
        rows.append(result)

    columns = (
        ["model"]
        + PROBABILITY_METRICS
        + THRESHOLD_METRICS
        + [
            "true_negatives",
            "false_positives",
            "false_negatives",
            "true_positives",
        ]
    )

    return (
        pd.DataFrame(rows)[columns]
        .sort_values("roc_auc", ascending=False)
        .reset_index(drop=True)
    )




def compare_cv_and_test(cv_comparison, test_results):
    """Align cross-validation and test performance for the same models."""

    cv_columns = cv_comparison[
        ["model", "mean_cv_validation_roc_auc"]
    ].rename(columns={"mean_cv_validation_roc_auc": "cv_roc_auc"})

    test_columns = test_results[
        ["model", "roc_auc"]
    ].rename(columns={"roc_auc": "test_roc_auc"})

    comparison = cv_columns.merge(test_columns, on="model", how="inner")

    # Confirm that no model was lost or duplicated by the merge
    assert len(comparison) == len(cv_columns) == len(test_columns)

    comparison["cv_test_gap"] = (
        comparison["cv_roc_auc"] - comparison["test_roc_auc"]
    )

    return comparison.sort_values(
        "test_roc_auc", ascending=False
    ).reset_index(drop=True)




def plot_roc_curves(models, X, y, figsize=(7, 6)):
    """Plot test-set ROC curves for every fitted model."""

    validate_evaluation_inputs(models, X, y)

    figure, ax = plt.subplots(figsize=figsize)

    for name, model in models.items():
        probabilities = predict_probabilities(model, X)

        false_positive_rate, true_positive_rate, _ = roc_curve(
            y, probabilities
        )

        area = roc_auc_score(y, probabilities)

        ax.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{name} (AUC = {area:.4f})"
        )

    # Reference line for a model with no discriminative ability
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Test-Set ROC Curves")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()

    return ax




def plot_precision_recall_curves(models, X, y, figsize=(7, 6)):
    """Plot test-set precision-recall curves for every fitted model."""

    validate_evaluation_inputs(models, X, y)

    figure, ax = plt.subplots(figsize=figsize)

    for name, model in models.items():
        probabilities = predict_probabilities(model, X)

        precision, recall, _ = precision_recall_curve(y, probabilities)
        area = average_precision_score(y, probabilities)

        ax.plot(recall, precision, label=f"{name} (AP = {area:.4f})")

    # Reference line for a model that predicts the base rate for every booking
    base_rate = float(np.mean(y))

    ax.axhline(
        base_rate,
        linestyle="--",
        color="grey",
        linewidth=1,
        label=f"No-skill baseline ({base_rate:.2%})"
    )

    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title("Test-Set Precision-Recall Curves")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()

    return ax




def make_confusion_table(model, X, y, threshold=DEFAULT_THRESHOLD):
    """Return a labeled confusion matrix for one fitted model."""

    probabilities = predict_probabilities(model, X)
    predictions = (probabilities >= threshold).astype(int)

    matrix = confusion_matrix(y, predictions, labels=[0, 1])

    return pd.DataFrame(
        matrix,
        index=pd.Index(
            ["Actual: Not canceled", "Actual: Canceled"],
            name=""
        ),
        columns=pd.Index(
            ["Predicted: Not canceled", "Predicted: Canceled"],
            name=""
        )
    )




def get_feature_names(pipeline):
    """Return the transformed predictor names produced by the preprocessor."""

    preprocessor = pipeline.named_steps["preprocessor"]

    return preprocessor.get_feature_names_out()




def get_model_importance(pipeline, top_n=20):
    """Return the most influential transformed predictors for one pipeline."""

    model = pipeline.named_steps["model"]
    names = get_feature_names(pipeline)

    # Tree-based models expose impurity-based importances
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
        value_column = "importance"

    # Linear models expose signed coefficients on the standardized scale
    elif hasattr(model, "coef_"):
        values = np.asarray(model.coef_).ravel()
        value_column = "coefficient"

    else:
        raise TypeError(
            f"{type(model).__name__} exposes neither "
            "`feature_importances_` nor `coef_`."
        )

    # Confirm that the values line up with the transformed predictor names
    assert len(values) == len(names), (len(values), len(names))

    importance = pd.DataFrame({
        "feature": names,
        value_column: values,
    })

    importance["magnitude"] = importance[value_column].abs()

    return (
        importance
        .sort_values("magnitude", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )




def get_permutation_importance(
    pipeline,
    X,
    y,
    random_state,
    n_repeats=5,
    n_jobs=1,
    top_n=20
):
    """Return permutation importances measured on the original predictors."""

    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        pipeline,
        X,
        y,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs
    )

    importance = pd.DataFrame({
        "feature": X.columns,
        "mean_auc_drop": result.importances_mean,
        "std_auc_drop": result.importances_std,
    })

    return (
        importance
        .sort_values("mean_auc_drop", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )




def threshold_sweep(model, X, y, thresholds=None):
    """Describe how threshold-based metrics change across candidate cutoffs."""

    if thresholds is None:
        thresholds = np.round(np.arange(0.20, 0.85, 0.05), 2)

    probabilities = predict_probabilities(model, X)

    rows = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        true_negatives, false_positives, false_negatives, true_positives = (
            confusion_matrix(y, predictions, labels=[0, 1]).ravel()
        )

        rows.append({
            "threshold": float(threshold),
            "accuracy": accuracy_score(y, predictions),
            "sensitivity": recall_score(y, predictions, zero_division=0),
            "specificity": (
                true_negatives / (true_negatives + false_positives)
            ),
            "precision": precision_score(y, predictions, zero_division=0),
            "f1": f1_score(y, predictions, zero_division=0),
        })

    return pd.DataFrame(rows)
