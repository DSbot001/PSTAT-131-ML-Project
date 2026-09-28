# Predicting Hotel Booking Cancellations

Final project for PSTAT 131 at UC Santa Barbara. The goal is to predict whether a hotel
booking will be canceled using only information available when the reservation is made.

[View the report](https://htmlpreview.github.io/?https://github.com/DSbot001/PSTAT-131-ML-Project/blob/main/hotel_project.html) ·
[Codebook](codebook.md)

## Data

The [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
dataset from Kaggle covers bookings at two hotels in Portugal with arrivals from July 2015 to
August 2017. The raw file has 119,390 rows. After removing 31,994 exact duplicates and 169
records with implausible guest counts, 87,227 bookings remain, of which 27.5% were canceled.
These were split into 69,781 training and 17,446 test observations.

## Approach

`reservation_status` and `reservation_status_date` directly reveal the outcome, so they were
removed. `required_car_parking_spaces` and `booking_changes` can be updated after the booking
is made, so they were also left out to keep the models honest about what is known at booking
time.

Five models were tuned with cross-validation on the training set:

- Elastic net logistic regression
- Pruned decision tree
- Random forest
- XGBoost
- Regularized QDA

## Results

Performance on the held-out test set at a 0.5 threshold:

| Model | ROC AUC | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|
| XGBoost | 0.9015 | 0.8389 | 0.6391 | 0.9148 |
| Random Forest | 0.9011 | 0.8390 | 0.6362 | 0.9160 |
| Pruned Decision Tree | 0.8773 | 0.8220 | 0.6129 | 0.9015 |
| Elastic Net Logistic | 0.8500 | 0.8007 | 0.5083 | 0.9117 |
| Regularized QDA | 0.8246 | 0.7174 | 0.8182 | 0.6791 |

XGBoost and random forest performed about the same and clearly beat the other three models.
The most important predictors for XGBoost were country, booking agent, and lead time.

The main limitations are that the train/test split is random rather than chronological, and
that the 0.5 threshold may not reflect how a hotel would weigh missed cancellations against
false alarms.

## Project Structure

```
├── hotel_project.qmd     # full analysis
├── hotel_project.html    # rendered report
├── codebook.md           # variable definitions
├── data_memo.qmd         # project proposal
├── data/                 # raw data
├── src/                  # feature engineering, preprocessing, training, evaluation
└── saved_models/         # fitted models
```
