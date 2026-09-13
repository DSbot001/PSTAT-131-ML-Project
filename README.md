# Predicting Hotel Booking Cancellations

Course project for PSTAT 131. Five classification models are compared on their ability to
predict, **at the moment a reservation is made**, whether that booking will later be
canceled. Variables that only become known after the reservation is entered are identified
and excluded, so the models stay usable for their stated purpose.

Full analysis: [`hotel_project.qmd`](hotel_project.qmd) · Variable definitions:
[`codebook.md`](codebook.md)

## Data

[Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
(Kaggle) — two hotels in Portugal, arrivals July 2015 to August 2017. After removing exact
duplicates and implausible guest counts, 87,227 bookings remain (69,781 train / 17,446 test,
27.5% canceled).

## Results

Test set, threshold 0.50:

| Model | ROC AUC | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|
| XGBoost | **0.9015** | 0.8389 | 0.6391 | 0.9148 |
| Random Forest | 0.9011 | 0.8390 | 0.6362 | 0.9160 |
| Pruned Decision Tree | 0.8773 | 0.8220 | 0.6129 | 0.9015 |
| Elastic Net Logistic | 0.8500 | 0.8007 | 0.5083 | 0.9117 |
| Regularized QDA | 0.8246 | 0.7174 | 0.8182 | 0.6791 |

The two ensembles differ by 0.0004 in test AUC, within sampling error — the result supports
the ensemble approach as a class rather than either model individually.

## Repository

```
hotel_project.qmd         Full analysis and report
codebook.md               Every column: definition, levels, modeling status
data_memo.qmd             Initial project proposal
data/                     Raw data
src/                      Feature engineering, preprocessing, training, evaluation
saved_models/             Fitted hyperparameter searches
```

## Running it

```bash
pip install -r requirements.txt
quarto render hotel_project.qmd
```

Fitted models are loaded from `saved_models/` by default; set `RETRAIN_MODELS = True` in the
setup chunk to refit from scratch. Working through the document interactively rather than
rendering it requires a fresh kernel — the cleaning chunks modify `hotel_clean` in place.

Requires Python 3.11+ and Quarto.
