# Hotel Booking Cancellation Project: Codebook

## Scope

This codebook describes the original Hotel Booking Demand data and the variables used in the engineered modeling dataset. Each row represents one hotel booking. The response is `is_canceled`, where 1 indicates a canceled booking and 0 indicates a booking that was not canceled.

The raw data contain 119,390 bookings from two Portuguese hotels, with arrivals from 2015 to 2017. Exact duplicate rows are removed before the train test split. After cleaning, 87,227 bookings remain (training 69,781, test 17,446). The final modeling data therefore represent cleaned booking records rather than the original file without preprocessing.

## Raw variables

| Variable | Description | Type | Modeling notes |
|---|---|---|---|
| `hotel` | Hotel type: City Hotel or Resort Hotel | Categorical | Included |
| `is_canceled` | Whether the booking was canceled: 0 = not canceled, 1 = canceled | Binary response | Target variable |
| `lead_time` | Number of days between booking and arrival | Numeric | Replaced by `log_lead_time` |
| `arrival_date_year` | Arrival year: 2015, 2016, 2017 | Numeric | Included |
| `arrival_date_month` | Arrival month as an English month name, January through December | Categorical | Not included. Used to create cyclical month features |
| `arrival_date_week_number` | Arrival week number | Numeric | Not selected for the primary model |
| `arrival_date_day_of_month` | Day of the month of arrival | Numeric | Included |
| `stays_in_weekend_nights` | Weekend nights booked | Count | Used to calculate stay length and weekend ratio |
| `stays_in_week_nights` | Weeknight stays booked | Count | Used to calculate stay length |
| `adults` | Number of adults | Count | Included |
| `children` | Number of children | Count | Included; missing values are imputed in preprocessing |
| `babies` | Number of babies | Count | Included; implausible values are screened during cleaning |
| `meal` | Meal package booked: BB = bed and breakfast; HB = half board; FB = full board; SC = self catering, no meal package; Undefined = no package recorded | Categorical | Included |
| `country` | Country of origin as an ISO 3166-1 alpha-3 code (177 levels; for example 'PRT', 'GBR', 'FRA'); missing values coded as "Unknown Country" | Categorical | Included as a high-cardinality feature; rare levels are pooled |
| `market_segment` | Market segment through which the booking was made: Aviation; Complementary; Corporate; Direct; Groups; Offline TA/TO; Online TA; Undefined. TA denotes travel agent, TO tour operator | Categorical | Included |
| `distribution_channel` | Distribution channel used for the booking: Corporate; Direct; GDS = global distribution system; TA/TO; Undefined | Categorical | Included |
| `is_repeated_guest` | Whether the guest has a previous booking at the property: 0 = no, 1 = yes | Binary | Included |
| `previous_cancellations` | Number of previous canceled bookings | Count | Included and also grouped into a categorical bucket |
| `previous_bookings_not_canceled` | Number of previous non-canceled bookings | Count | Included; also used to create `has_prior_non_canceled_booking` |
| `reserved_room_type` | Room type reserved, as an anonymized letter code: A, B, C, D, E, F, G, H, L. The letters carry no ordering | Categorical | Included |
| `assigned_room_type` | Room type ultimately assigned, as an anonymized letter code: A, B, C, D, E, F, G, H, I, K, L | Categorical | Excluded because it may be known only after booking or during fulfillment |
| `booking_changes` | Number of changes made to the booking | Count | Excluded from the primary booking-time model |
| `deposit_type` | Deposit terms: No Deposit = no deposit taken; Non Refund = deposit equal to the full stay cost, not refundable; Refundable = deposit below the full cost, refundable | Categorical | Included |
| `agent` | Travel agency identifier, an anonymized numeric ID treated as a label (333 levels); bookings with no agency coded as "No Agent" | Categorical | Included as a high-cardinality feature; missing values are represented as an explicit level and rare levels are pooled |
| `company` | Company identifier associated with the booking | Categorical | Replaced by `has_company` and excluded in raw form |
| `days_in_waiting_list` | Days the booking remained on a waiting list | Count | Excluded: not available at booking time |
| `customer_type` | Customer category: Contract = tied to an allotment or contract; Group = associated with a group; Transient = individual booking, not part of a group or contract; Transient-Party = transient booking linked to at least one other transient booking | Categorical | Included |
| `adr` | Average daily rate | Numeric | Replaced by cleaned `adr_adj` |
| `required_car_parking_spaces` | Number of parking spaces requested | Count | Excluded from the primary model |
| `total_of_special_requests` | Number of special requests | Count | Included; also used to create `has_special_request` |
| `reservation_status` | Final reservation status: Check-Out; Canceled; No-Show | Categorical | Removed as response leakage |
| `reservation_status_date` | Date of the final reservation status | Date | Removed as response leakage |



## Engineered variables

All indicator variables in this table are coded 0 = no, 1 = yes.

| Variable | Definition or purpose | Primary model use |
|---|---|---|
| `log_lead_time` | `log1p(lead_time)`; reduces the right skew in lead time | Included |
| `arrival_month_num` | Numeric month index from 1 to 12 | Used to construct cyclical features; not included  |
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
| `has_booking_change` | Indicator for at least one booking change | excluded from the primary model |
| `has_parking_request` | Indicator for at least one parking-space request | excluded from the primary model |




## Evaluation metrics

| Metric | Definition |
|---|---|
| ROC AUC | Ranking-based discrimination between canceled and non-canceled bookings |
| Accuracy | Proportion of all bookings classified correctly |
| Sensitivity | Proportion of actual cancellations detected; also called recall or true-positive rate |
| Specificity | Proportion of non-canceled bookings correctly identified |
| Precision | Proportion of predicted cancellations that are actual cancellations |
| F1 score | Harmonic mean of precision and sensitivity |
| Train validation AUC gap | Training AUC minus mean validation AUC |
| CV test AUC gap | Mean cross-validation AUC minus test-set AUC |