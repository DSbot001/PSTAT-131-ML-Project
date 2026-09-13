# Hotel Booking Cancellation Project: Codebook

## Scope

This codebook describes the original Hotel Booking Demand data and the variables used in the engineered modeling dataset. Each row represents one hotel booking. The response is `is_canceled`, where 1 indicates a canceled booking and 0 indicates a booking that was not canceled.

The raw data contain 119,390 bookings from two Portuguese hotels, with arrivals from 2015â€“2017. Exact duplicate rows are removed before the trainâ€“test split. The final modeling data therefore represent cleaned booking records rather than the original file without preprocessing.

## Raw variables

| Variable | Description | Type | Modeling notes |
|---|---|---|---|
| `hotel` | Hotel type: City Hotel or Resort Hotel | Categorical | Included |
| `is_canceled` | Whether the booking was canceled | Binary response | Target variable |
| `lead_time` | Number of days between booking and arrival | Numeric | Replaced by `log_lead_time` |
| `arrival_date_year` | Arrival year | Numeric | Included |
| `arrival_date_month` | Arrival month | Categorical | Used to create cyclical month features |
| `arrival_date_week_number` | Arrival week number | Numeric | Not selected for the primary model |
| `arrival_date_day_of_month` | Day of the month of arrival | Numeric | Included |
| `stays_in_weekend_nights` | Weekend nights booked | Count | Used to calculate stay length and weekend ratio |
| `stays_in_week_nights` | Weeknight stays booked | Count | Used to calculate stay length |
| `adults` | Number of adults | Count | Included |
| `children` | Number of children | Count | Included; missing values are imputed in preprocessing |
| `babies` | Number of babies | Count | Included; implausible values are screened during cleaning |
| `meal` | Meal plan selected | Categorical | Included |
| `country` | Country of origin of the guest | Categorical | Included as a high-cardinality feature; rare levels are pooled |
| `market_segment` | Market segment through which the booking was made | Categorical | Included |
| `distribution_channel` | Distribution channel used for the booking | Categorical | Included |
| `is_repeated_guest` | Whether the guest is a repeat guest | Binary | Included |
| `previous_cancellations` | Number of previous canceled bookings | Count | Included and also grouped into a categorical bucket |
| `previous_bookings_not_canceled` | Number of previous non-canceled bookings | Count | Included; also used to create `has_prior_non_canceled_booking` |
| `reserved_room_type` | Room type reserved | Categorical | Included |
| `assigned_room_type` | Room type ultimately assigned | Categorical | Excluded because it may be known only after booking or during fulfillment |
| `booking_changes` | Number of changes made to the booking | Count | Excluded from the primary booking-time model |
| `deposit_type` | Deposit arrangement | Categorical | Included |
| `agent` | Travel agent identifier | Categorical | Included as a high-cardinality feature; missing values are represented as an explicit level and rare levels are pooled |
| `company` | Company identifier associated with the booking | Categorical | Replaced by `has_company` and excluded in raw form |
| `days_in_waiting_list` | Days the booking remained on a waiting list | Count | Excluded from the primary model |
| `customer_type` | Customer category | Categorical | Included |
| `adr` | Average daily rate | Numeric | Replaced by cleaned `adr_adj` |
| `required_car_parking_spaces` | Number of parking spaces requested | Count | Reserved for sensitivity analysis; excluded from the primary model |
| `total_of_special_requests` | Number of special requests | Count | Included; also used to create `has_special_request` |
| `reservation_status` | Final reservation status | Categorical | Removed as response leakage |
| `reservation_status_date` | Date of the final reservation status | Date | Removed as response leakage |

## Engineered variables

| Variable | Definition or purpose | Primary model use |
|---|---|---|
| `log_lead_time` | `log1p(lead_time)`; reduces the right skew in lead time | Included |
| `arrival_month_num` | Numeric month index from 1 to 12 | Used to construct cyclical features; not included directly |
| `arrival_month_sin` | Sine transformation of the month angle | Included |
| `arrival_month_cos` | Cosine transformation of the month angle | Included |
| `total_nights_adj` | Adjusted total stay length; questionable zero-night records remain missing | Included |
| `weekend_night_ratio` | Weekend nights divided by adjusted total nights | Included |
| `adr_adj` | Adjusted ADR used in place of raw `adr` when the raw value is questionable | Included |
| `has_company` | Indicator that a company identifier is present | Included |
| `has_prior_non_canceled_booking` | 1 if `previous_bookings_not_canceled` is greater than 0 | Included |
| `has_special_request` | 1 if at least one special request is recorded | Included |
| `previous_cancellations_bucket` | Previous cancellations grouped into `0`, `1`, and `2+` | Included as categorical |
| `is_family_booking` | Indicates at least one child or baby; missing status is preserved when the counts are unknown | Included |
| `non_refund_groups` | Indicator for a non-refundable deposit and the Groups market segment | Included |
| `non_refund_offline_ta_to` | Indicator for a non-refundable deposit and the Offline TA/TO segment | Included |
| `has_booking_change` | Indicator for at least one booking change | Sensitivity analysis only; excluded from the primary model |
| `has_parking_request` | Indicator for at least one parking-space request | Sensitivity analysis only; excluded from the primary model |

## Primary modeling feature set

The primary design matrix contains 31 predictors divided into four preprocessing groups:

### Numeric and count predictors

`log_lead_time`, `arrival_date_year`, `arrival_date_day_of_month`, `arrival_month_sin`, `arrival_month_cos`, `total_nights_adj`, `weekend_night_ratio`, `adults`, `children`, `babies`, `previous_cancellations`, `previous_bookings_not_canceled`, `adr_adj`, and `total_of_special_requests`.

### Binary predictors

`is_repeated_guest`, `has_company`, `has_prior_non_canceled_booking`, `has_special_request`, `is_family_booking`, `non_refund_groups`, and `non_refund_offline_ta_to`.

### Ordinary categorical predictors

`hotel`, `meal`, `market_segment`, `distribution_channel`, `reserved_room_type`, `deposit_type`, `customer_type`, and `previous_cancellations_bucket`.

### High-cardinality categorical predictors

`agent` and `country`. Missing values are handled explicitly, and infrequent levels are pooled during preprocessing.

## Preprocessing and evaluation notes

- Numeric variables are median-imputed; scale-sensitive models additionally standardize them.
- Binary variables are imputed using the most frequent training-fold value.
- Ordinary categorical variables use most-frequent imputation and one-hot encoding.
- `agent` and `country` use one-hot encoding with rare-level pooling and unknown-level handling.
- Preprocessing is fitted within each training fold to avoid using information from validation or test data.
- `reservation_status` and `reservation_status_date` are excluded because they describe the outcome or information recorded after the booking decision.
- Native tree importance describes model reliance on a feature, not the direction or causality of its relationship with cancellation.
- Permutation importance is reported as the decrease in test ROC AUC after shuffling one predictor while keeping the fitted model fixed.
- Confusion-matrix metrics use a probability threshold of 0.50 unless otherwise stated.

## Evaluation metrics

| Metric | Definition |
|---|---|
| ROC AUC | Ranking-based discrimination between canceled and non-canceled bookings |
| Accuracy | Proportion of all bookings classified correctly |
| Sensitivity | Proportion of actual cancellations detected; also called recall or true-positive rate |
| Specificity | Proportion of non-canceled bookings correctly identified |
| Precision | Proportion of predicted cancellations that are actual cancellations |
| F1 score | Harmonic mean of precision and sensitivity |
| Trainâ€“validation AUC gap | Training AUC minus mean validation AUC |
| CVâ€“test AUC gap | Mean cross-validation AUC minus test-set AUC |