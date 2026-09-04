import numpy as np
import pandas as pd


MONTH_TO_NUMBER = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}



def _normalize_agent_id(series):
    """Convert equivalent agent IDs such as 240 and 240.0 to '240'."""

    normalized = (
        series
        .fillna("No Agent")
        .astype(str)
        .str.strip()
    )

    numeric = pd.to_numeric(series, errors="coerce")

    integer_like = (
        numeric.notna()
        & np.isclose(numeric % 1, 0)
    )

    normalized.loc[integer_like] = (
        numeric.loc[integer_like]
        .astype("int64")
        .astype(str)
    )

    return normalized.astype(object)





def engineer_features(df):
    """Add deterministic booking-level features without selecting predictors."""
    data = df.copy()

    if "is_canceled" in data.columns:
        raise ValueError(
            "Pass predictor columns only; do not include `is_canceled`."
        )

    leakage_columns = {
        "reservation_status",
        "reservation_status_date"
    }

    unexpected_columns = leakage_columns.intersection(data.columns)

    if unexpected_columns:
        raise ValueError(
            "Leakage columns should be removed during data cleaning: "
            f"{sorted(unexpected_columns)}"
        )

    # Lead-time transformation
    if data["lead_time"].lt(0).any():
        raise ValueError(
            "`lead_time` must be non-negative before applying log1p."
        )

    data["log_lead_time"] = np.log1p(data["lead_time"])

    # Alternative month representations
    month_num = data["arrival_date_month"].map(MONTH_TO_NUMBER)

    unknown_months = data.loc[
        month_num.isna()
        & data["arrival_date_month"].notna(),
        "arrival_date_month"
    ].unique()

    if len(unknown_months) > 0:
        raise ValueError(
            f"Unrecognized month values: {unknown_months.tolist()}"
        )

    data["arrival_month_num"] = month_num.astype("float64")

    month_angle = (
        2 * np.pi * (month_num - 1) / 12
    )

    data["arrival_month_sin"] = np.sin(month_angle)
    data["arrival_month_cos"] = np.cos(month_angle)

    # Use the adjusted total so both variables are missing
    # for the same questionable-stay observations
    ratio_denominator = data["total_nights_adj"].where(
        data["total_nights_adj"] > 0
    )

    data["weekend_night_ratio"] = (
        data["stays_in_weekend_nights"]
        .div(ratio_denominator)
    )

    # Threshold patterns identified during EDA
    data["has_prior_non_canceled_booking"] = (
        data["previous_bookings_not_canceled"] > 0
    ).astype("int8")

    data["has_special_request"] = (
        data["total_of_special_requests"] > 0
    ).astype("int8")

    data["previous_cancellations_bucket"] = pd.cut(
        data["previous_cancellations"],
        bins=[-np.inf, 0, 1, np.inf],
        labels=["0", "1", "2+"]
    ).astype(object)

    # Preserve uncertainty when `children` or `babies` is missing
    known_family = (
        data["children"].gt(0)
        | data["babies"].gt(0)
    )

    unknown_family = (
        data["children"].isna()
        | data["babies"].isna()
    ) & ~known_family

    data["is_family_booking"] = (
        known_family
        .astype("float64")
        .mask(unknown_family)
    )

    # Targeted interaction indicators
    data["non_refund_groups"] = (
        data["deposit_type"].eq("Non Refund")
        & data["market_segment"].eq("Groups")
    ).astype("int8")

    data["non_refund_offline_ta_to"] = (
        data["deposit_type"].eq("Non Refund")
        & data["market_segment"].eq("Offline TA/TO")
    ).astype("int8")

    # Created for later prediction-time sensitivity analyses
    data["has_booking_change"] = (
        data["booking_changes"] > 0
    ).astype("int8")

    data["has_parking_request"] = (
        data["required_car_parking_spaces"] > 0
    ).astype("int8")

    # Standardize identifier formats
    data["agent"] = _normalize_agent_id(data["agent"])

    data["country"] = (
        data["country"]
        .fillna("Unknown Country")
        .astype(str)
        .str.strip()
        .astype(object)
    )

    # No raw or sensitivity columns are removed
    return data




def validate_engineered_features(original, engineered):
    """Validate that feature engineering preserves rows and raw predictors."""

    expected_new_columns = {
        "log_lead_time",
        "arrival_month_num",
        "arrival_month_sin",
        "arrival_month_cos",
        "weekend_night_ratio",
        "has_prior_non_canceled_booking",
        "has_special_request",
        "previous_cancellations_bucket",
        "is_family_booking",
        "non_refund_groups",
        "non_refund_offline_ta_to",
        "has_booking_change",
        "has_parking_request",
    }

    assert len(engineered) == len(original)
    assert engineered.index.equals(original.index)

    removed_columns = (
        set(original.columns)
        - set(engineered.columns)
    )

    missing_new_columns = (
        expected_new_columns
        - set(engineered.columns)
    )

    assert not removed_columns, (
        f"Feature engineering removed columns: {removed_columns}"
    )

    assert not missing_new_columns, (
        f"Missing engineered columns: {missing_new_columns}"
    )

    assert "is_canceled" not in engineered.columns

    binary_columns = [
        "has_prior_non_canceled_booking",
        "has_special_request",
        "is_family_booking",
        "non_refund_groups",
        "non_refund_offline_ta_to",
        "has_booking_change",
        "has_parking_request",
    ]

    for column in binary_columns:
        observed_values = set(
            engineered[column].dropna().unique()
        )

        assert observed_values <= {0, 1}, (
            column,
            observed_values,
        )

    bucket_levels = set(
        engineered[
            "previous_cancellations_bucket"
        ].dropna().unique()
    )

    assert bucket_levels <= {"0", "1", "2+"}