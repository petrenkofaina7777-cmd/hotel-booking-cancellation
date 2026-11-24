# Predicting hotel booking cancellations

Binary classification: will a hotel booking be cancelled? Built with scikit-learn on the Hotel Booking Demand data, with a focus on **honest evaluation** – removing duplicate rows and a leaky feature changes the picture noticeably.

**Data:** [Hotel Booking Demand (Kaggle)](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) – 119,390 bookings of a city and a resort hotel in Portugal, 2015–2017 (Antonio, Almeida & Nunes, 2019).
Download `hotel_bookings.csv` and put it next to the notebook.

## What is in the notebook
- Cleaning: missing values, **28.7 % exact duplicates removed**, `reservation_status` excluded (leaks the outcome)
- **Leakage check:** `assigned_room_type` is only known at check-in – bookings with a changed room are cancelled in 4.8 % of cases vs. 31.9 % – so it is dropped
- Exploratory analysis of cancellation rates and lead time
- Leak-free `Pipeline` (scaling + one-hot encoding), stratified 80/20 split
- Majority baseline, Decision Tree, KNN, Random Forest and Gradient Boosting; `GridSearchCV` on the training set only
- Evaluation: accuracy, precision, recall, F1, ROC-AUC, ROC curves, confusion matrix
- Permutation importance and a decision-threshold analysis

## Results (de-duplicated data, test set of 17,017 bookings)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Majority baseline | 0.721 | 0.000 | 0.000 | 0.000 | 0.500 |
| Decision Tree (tuned) | 0.791 | 0.695 | 0.448 | 0.545 | 0.797 |
| KNN (tuned) | 0.782 | 0.688 | 0.396 | 0.503 | 0.775 |
| **Random Forest** | **0.806** | 0.735 | 0.475 | 0.577 | **0.832** |
| Gradient Boosting (tuned) | 0.803 | 0.715 | 0.487 | 0.579 | 0.826 |

## Key findings
- **Duplicates inflate the scores:** with the duplicates left in, the same Decision Tree / KNN reach ~81 % accuracy and 0.86–0.87 ROC-AUC; without them 0.79 / 0.78 and 0.79 / 0.77 – only 6–7 percentage points above the majority baseline (72.1 %).
- Most important predictors: **lead time, market segment, number of special requests**, customer type, ADR and previous cancellations.
- At the default threshold the Random Forest finds 47.5 % of cancellations; at threshold 0.3 it finds 72.5 % with 56 % precision.
- Limitations: random (not chronological) split, two Portuguese hotels, non-refundable-deposit pattern looks like a data artefact.

## Run it
```bash
pip install -r requirements.txt
jupyter notebook hotel_cancellation_classification.ipynb
```
(The full run takes roughly 15–20 minutes on a single CPU core, mainly because of the grid searches and the KNN models.)
