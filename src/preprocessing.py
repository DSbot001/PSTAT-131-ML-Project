from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_COLUMNS = [
    "log_lead_time",
    "arrival_date_year",
    "arrival_date_day_of_month",
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


BINARY_COLUMNS = [
    "is_repeated_guest",
    "has_company",
    "has_prior_non_canceled_booking",
    "has_special_request",
    "is_family_booking",
    "non_refund_groups",
    "non_refund_offline_ta_to",
]


CATEGORICAL_COLUMNS = [
    "hotel",
    "arrival_date_month",
    "meal",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
    "previous_cancellations_bucket",
]


HIGH_CARDINALITY_COLUMNS = [
    "agent",
    "country",
]


MODEL_COLUMNS = (
    NUMERIC_COLUMNS
    + BINARY_COLUMNS
    + CATEGORICAL_COLUMNS
    + HIGH_CARDINALITY_COLUMNS
)


def validate_feature_columns(df):
    """Check that all model predictors exist after feature engineering."""

    missing_columns = sorted(
        set(MODEL_COLUMNS) - set(df.columns)
    )

    if missing_columns:
        raise KeyError(
            f"Missing model columns: {missing_columns}"
        )


def make_preprocessor(scale_numeric=True):
    """Create preprocessing steps to be fitted within each training fold."""

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipeline = Pipeline(numeric_steps)

    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    high_cardinality_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=0.005,
                sparse_output=False
            )
        )
    ])

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_COLUMNS
            ),
            (
                "binary",
                binary_pipeline,
                BINARY_COLUMNS
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_COLUMNS
            ),
            (
                "high_cardinality",
                high_cardinality_pipeline,
                HIGH_CARDINALITY_COLUMNS
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )