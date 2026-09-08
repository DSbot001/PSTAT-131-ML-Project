from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.preprocessing import make_preprocessor
from xgboost import XGBClassifier

from sklearn.discriminant_analysis import (
    QuadraticDiscriminantAnalysis
)
from sklearn.preprocessing import FunctionTransformer


CV_FOLDS = 5

# Use ROC AUC for model selection while retaining threshold-based diagnostics


SCORING = {
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
    "sensitivity": "recall",
    "specificity": make_scorer(recall_score, pos_label=0),
}


def make_cv(random_state):
    """Create the shared stratified folds used for all model comparisons."""

    return StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=random_state
    )


def make_elastic_net_pipeline(random_state):
    """Create the preprocessing and Elastic Net logistic pipeline."""

    model = LogisticRegression(
        solver="saga",
        max_iter=1000,
        tol=1e-3,
        random_state=random_state
    )

    return Pipeline([
        ("preprocessor", make_preprocessor(scale_numeric=True)),
        ("model", model)
    ])


def make_elastic_net_search(random_state, n_jobs=4):
    """Create the initial Elastic Net cross-validation search."""

    pipeline = make_elastic_net_pipeline(random_state)

    param_grid = {
        "model__C": [0.01, 0.1, 1, 10, 100, 1000],
        "model__l1_ratio": [0.0, 0.2, 0.5, 1.0],
    }

    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=SCORING,
        refit="roc_auc",
        cv=make_cv(random_state),
        n_jobs=n_jobs,
        pre_dispatch=n_jobs,
        return_train_score=True,
        verbose=1
    )


def make_decision_tree_pipeline(random_state):
    """Create the preprocessing and decision tree pipeline."""

    model = DecisionTreeClassifier(
        random_state=random_state
    )

    return Pipeline([
        (
            "preprocessor",
            make_preprocessor(scale_numeric=False)
        ),
        ("model", model)
    ])


def make_decision_tree_search(
    random_state,
    n_jobs=4
):
    """Create the pruned decision tree cross-validation search."""

    pipeline = make_decision_tree_pipeline(
        random_state
    )

    param_grid = {
        "model__max_depth": [
            4,
            6,
            8,
            None
        ],
        "model__min_samples_leaf": [
            20,
            100,
            300
        ],
        "model__ccp_alpha": [
            0.0,
            0.00001,
            0.0001,
            0.001
        ],
    }

    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=SCORING,
        refit="roc_auc",
        cv=make_cv(random_state),
        n_jobs=n_jobs,
        pre_dispatch=n_jobs,
        return_train_score=True,
        verbose=1
    )




def make_random_forest_pipeline(random_state):
    """Create the preprocessing and random forest pipeline."""

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=1
    )

    return Pipeline([
        (
            "preprocessor",
            make_preprocessor(scale_numeric=False)
        ),
        ("model", model)
    ])


def make_random_forest_search(
    random_state,
    n_jobs=4
):
    """Create the random forest cross-validation search."""

    pipeline = make_random_forest_pipeline(
        random_state
    )

    param_grid = {
        "model__max_depth": [
            10,
            14,
            18,
            22
        ],
        "model__min_samples_leaf": [
            5,
            10,
            20,
            50
        ],
        "model__max_features": [
            "sqrt",
            0.3
        ],
    }

    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=SCORING,
        refit="roc_auc",
        cv=make_cv(random_state),
        n_jobs=n_jobs,
        pre_dispatch=n_jobs,
        return_train_score=True,
        verbose=1
    )



def make_xgboost_pipeline(random_state):
    """Create the preprocessing and XGBoost pipeline."""

    model = XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        eval_metric="auc",
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=1,
        verbosity=0
    )

    return Pipeline([
        (
            "preprocessor",
            make_preprocessor(scale_numeric=False)
        ),
        ("model", model)
    ])


def make_xgboost_search(
    random_state,
    n_jobs=4
):
    """Create the XGBoost cross-validation search."""

    pipeline = make_xgboost_pipeline(
        random_state
    )

    param_grid = {
        "model__n_estimators": [
            400,
            800
        ],
        "model__max_depth": [
            3,
            5,
            7
        ],
        "model__learning_rate": [
            0.03,
            0.06
        ],
        "model__min_child_weight": [
            3,
            10
        ],
        "model__subsample": [
            0.8
        ],
        "model__colsample_bytree": [
            0.7,
            0.9
        ],
    }

    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=SCORING,
        refit="roc_auc",
        cv=make_cv(random_state),
        n_jobs=n_jobs,
        pre_dispatch=n_jobs,
        return_train_score=True,
        verbose=1
    )



def to_dense_matrix(X):
    """Convert a sparse preprocessing output to a dense matrix."""

    if hasattr(X, "toarray"):
        return X.toarray()

    return X


def make_qda_pipeline():
    """Create the preprocessing and regularized QDA pipeline."""

    model = QuadraticDiscriminantAnalysis()

    return Pipeline([
        (
            "preprocessor",
            make_preprocessor(scale_numeric=True)
        ),
        (
            "to_dense",
            FunctionTransformer(
                to_dense_matrix,
                accept_sparse=True
            )
        ),
        ("model", model)
    ])


def make_qda_search(random_state, n_jobs=4):
    """Create the regularized QDA cross-validation search."""

    pipeline = make_qda_pipeline()

    param_grid = {
        "model__reg_param": [
            0.01,
            0.05,
            0.10,
            0.20,
            0.40,
            0.60,
            0.80
        ],
    }

    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=SCORING,
        refit="roc_auc",
        cv=make_cv(random_state),
        n_jobs=n_jobs,
        pre_dispatch=n_jobs,
        return_train_score=True,
        verbose=1
    )