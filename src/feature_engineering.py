import numpy as np
import pandas as pd



# Predictor columns that are known to leak the response variable and should be removed during data cleaning
PERMANENT_LEAKAGE_COLUMNS = [
    "reservation_status",
    "reservation_status_date",
]

# # Variables unavailable when the booking is initially created
POST_BOOKING_COLUMNS = [
    "assigned_room_type",
    "booking_changes",
    "days_in_waiting_list",
    "has_booking_change",
]


# Parking variables excluded because they completely separate the training outcome
SEPARATION_SUSPECT_COLUMNS = [
    "required_car_parking_spaces", # Original parking-space request count
    "has_parking_request",         # Binary indicator of any parking-space request 
]

# rows that have adr = 0 and total_nights = 0 are questionable stays and should be excluded from the training set
QUALITY_FLAG_COLUMNS = [
    "questionable_stay_data",
]


# Raw variables replaced by cleaned or more interpretable versions
SUPERSEDED_COLUMNS = [
    "adr",      # Replaced by `adr_adj`
    "company",  # Replaced by `has_company`
]



# Variables excluded from the primary booking-time design matrix
PRIMARY_INELIGIBLE_COLUMNS = (
    POST_BOOKING_COLUMNS
    + SEPARATION_SUSPECT_COLUMNS
    + QUALITY_FLAG_COLUMNS
    + SUPERSEDED_COLUMNS
)



# Convert month names into their calendar positions
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




"""Add deterministic booking-level features without selecting predictors."""

def engineer_features(df):
    
    data = df.copy()

    # Prevent the response variable from entering feature engineering
    if "is_canceled" in data.columns:
        raise ValueError(
            "Pass predictor columns only; do not include `is_canceled`."
        )

    # Confirm that permanent leakage was removed during data cleaning
    unexpected_columns = set(PERMANENT_LEAKAGE_COLUMNS).intersection(data.columns)

    if unexpected_columns:
        raise ValueError(
            "Leakage columns should be removed during data cleaning: "
            f"{sorted(unexpected_columns)}"
        )




    # Confirm that lead time is valid before applying the logarithm
    if data["lead_time"].lt(0).any():
        raise ValueError("`lead_time` must be non-negative before applying log1p.")

    # Compress the right-skewed lead-time distribution
    data["log_lead_time"] = np.log1p(
        data["lead_time"]
    )




    # Convert the month name into a numerical calendar position
    month_num = data["arrival_date_month"].map(MONTH_TO_NUMBER)
    data["arrival_month_num"] = month_num.astype("int8")

    # Convert month position into an angle on a twelve-month circle
    month_angle = (2 * np.pi * (month_num - 1) / 12)
    data["arrival_month_sin"] = np.sin(month_angle)
    data["arrival_month_cos"] = np.cos(month_angle)





    # Use adjusted total nights so questionable stays remain missing
    ratio_denominator = data["total_nights_adj"].where(data["total_nights_adj"] > 0)
    # Measure the proportion of the planned stay occurring on weekends
    data["weekend_night_ratio"] = (data["stays_in_weekend_nights"].div(ratio_denominator))




    # binary variable indicating whether the guest has any prior non-canceled booking
    data["has_prior_non_canceled_booking"] = (data["previous_bookings_not_canceled"] > 0).astype("int8")




    # Indicate whether the booking includes any special request
    data["has_special_request"] = (data["total_of_special_requests"] > 0).astype("int8")



    # Group previous cancellations into the EDA-supported levels 0, 1, and 2+
    data["previous_cancellations_bucket"] = pd.cut(
        data["previous_cancellations"],
        bins=[-np.inf, 0, 1, np.inf],
        labels=["0", "1", "2+"]
    ).astype(object)



    # Identify bookings known to include at least one child or baby
    known_family = (data["children"].gt(0)| data["babies"].gt(0))
    # Preserve missing family status when no positive count resolves the uncertainty
    unknown_family = (data["children"].isna()| data["babies"].isna()) & ~known_family




    # Create the family indicator while retaining genuinely unknown values
    data["is_family_booking"] = (known_family.astype("float64").mask(unknown_family))




    'Create intersection binary indicators for non-refundable bookings in specific market segments ------ Groups and Offline TA/TO.'

    data["non_refund_groups"] = (data["deposit_type"].eq("Non Refund")& data["market_segment"].eq("Groups")).astype("int8")

    data["non_refund_offline_ta_to"] = (data["deposit_type"].eq("Non Refund")& data["market_segment"].eq("Offline TA/TO")).astype("int8")



    # Create a post-booking indicator for later prediction-time sensitivity analysis
    data["has_booking_change"] = (
        data["booking_changes"] > 0
    ).astype("int8")



    # Create a parking indicator for complete-separation sensitivity analysis
    data["has_parking_request"] = (data["required_car_parking_spaces"] > 0).astype("int8")


    return data













def validate_engineered_features(original, engineered):
    """Validate that feature engineering preserves rows and raw predictors."""

    # List every column that `engineer_features()` should create
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


    # Confirm that feature engineering does not add or remove observations
    assert len(engineered) == len(original)


    # Confirm that the original row index remains unchanged
    assert engineered.index.equals(original.index)

    # Identify any original predictors accidentally removed by the function
    removed_columns = (set(original.columns) - set(engineered.columns))

    # Identify any expected engineered features that were not created
    missing_new_columns = (expected_new_columns - set(engineered.columns))


    # Stop if feature engineering removed any original predictor
    assert not removed_columns, (
        f"Feature engineering removed columns: {removed_columns}"
    )

    # Stop if feature engineering failed to create an expected feature
    assert not missing_new_columns, (
        f"Missing engineered columns: {missing_new_columns}"
    )

    # Confirm that the response variable is absent from the predictors
    assert "is_canceled" not in engineered.columns

    # List engineered indicators expected to contain only 0, 1, or missing
    binary_columns = [
        "has_prior_non_canceled_booking",
        "has_special_request",
        "is_family_booking",
        "non_refund_groups",
        "non_refund_offline_ta_to",
        "has_booking_change",
        "has_parking_request",
    ]

    # Confirm that every engineered indicator has valid binary values
    for column in binary_columns:
        observed_values = set(
            engineered[column].dropna().unique()
        )

        assert observed_values <= {0, 1}, (
            column,
            observed_values,
        )

    # Collect the observed previous-cancellation bucket labels
    bucket_levels = set(
        engineered[
            "previous_cancellations_bucket"
        ].dropna().unique()
    )

    # Confirm that no unintended cancellation bucket was created
    assert bucket_levels <= {"0", "1", "2+"}



