from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_engineering import PRIMARY_INELIGIBLE_COLUMNS






# Continuous and count predictors used by the engineered linear models
NUMERIC_COLUMNS = [
    "log_lead_time",
    "arrival_date_year",
    "arrival_date_day_of_month",
    "arrival_month_sin",
    "arrival_month_cos",
    "total_nights_adj",
    "weekend_night_ratio",
    "adults",
    "children",
    "babies",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "adr_adj",
    "total_of_special_requests",
]



# Binary predictors that do not require one-hot encoding
BINARY_COLUMNS = [
    "is_repeated_guest",
    "has_company",
    "has_prior_non_canceled_booking",
    "has_special_request",
    "is_family_booking",
    "non_refund_groups",
    "non_refund_offline_ta_to",
]


# Low-cardinality predictors encoded with ordinary one-hot encoding
CATEGORICAL_COLUMNS = [
    "hotel",
    "meal",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
    "previous_cancellations_bucket",
]



# High-cardinality predictors whose rare levels will be pooled
HIGH_CARDINALITY_COLUMNS = [
    "agent",
    "country",
]





# Combine all predictors included in the primary engineered design matrix
MODEL_COLUMNS = (
    NUMERIC_COLUMNS
    + BINARY_COLUMNS
    + CATEGORICAL_COLUMNS
    + HIGH_CARDINALITY_COLUMNS
)





def validate_feature_columns(df):
    """Check that all selected predictors exist and are eligible."""

    # Detect predictors accidentally listed in more than one preprocessing group
    duplicate_columns = sorted({
        column for column in MODEL_COLUMNS
        if MODEL_COLUMNS.count(column) > 1
    })

    if duplicate_columns:
        raise ValueError(
            f"Columns appear in multiple preprocessing groups: {duplicate_columns}"
        )

    # Confirm that every selected predictor exists after feature engineering
    missing_columns = sorted(set(MODEL_COLUMNS) - set(df.columns))

    if missing_columns:
        raise KeyError(f"Missing model columns: {missing_columns}")



    # Prevent leakage, sensitivity, and superseded variables from entering the primary model
    ineligible_columns = sorted(
        set(MODEL_COLUMNS) & set(PRIMARY_INELIGIBLE_COLUMNS)
    )

    if ineligible_columns:
        raise ValueError(
            f"Ineligible columns selected for the primary model: {ineligible_columns}"
        )










def make_preprocessor(scale_numeric=True):
    """Create preprocessing steps to be fitted within each training fold."""

    # Impute numerical variables with training-fold medians
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    # Standardize numerical variables for scale-sensitive models （Logistic Regression / KNN...）
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)



    # Impute missing binary values without changing their 0/1 interpretation
    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])




    # Impute and one-hot encode ordinary categorical predictors
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )),
    ])



    # Pool rare agent and country levels before one-hot encoding
    high_cardinality_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=0.002,
            sparse_output=False
        )),
    ])



    # Apply each pipeline only to its explicitly listed columns
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("binary", binary_pipeline, BINARY_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
            ("high_cardinality", high_cardinality_pipeline, HIGH_CARDINALITY_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )