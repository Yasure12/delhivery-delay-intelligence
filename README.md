
# Delhivery Delay Intelligence Capstone Project

## 1. Project Overview

This project uses synthetic shipment data to analyse delivery delays, predict whether new shipments may be delayed, and identify logistics hubs that need attention.

The project is designed as a demonstration and does not use real Delhivery operational data.

## 2. Project Objectives

- Clean and prepare shipment data.
- Explore delivery delays and important factors.
- Predict delivery time using regression models.
- Classify shipments as delayed or not delayed.
- Group logistics hubs based on their operational performance.
- Save a reusable machine learning model.
- Predict delays for new shipments and recommend actions.

## 3. Dataset

The synthetic dataset contains:

- 24 logistics hubs.
- 1,200 historical shipments.
- 48 new shipments for prediction.

The raw historical data contains duplicate records and missing values. These are handled during data preparation.

## 4. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Google Colab

## 5. Project Workflow

### Part A: Data Cleaning and EDA

- Removed duplicate records.
- Handled missing values.
- Corrected service type formatting.
- Created delivery delay labels.
- Explored delay patterns by service type, hub tier, and distance.

### Part B: Regression

Compared regression models to predict delivery time.

The two leading validation models were:

- Linear Regression: RMSE 4.89 hours.
- Gradient Boosting: RMSE 4.92 hours.

Linear Regression had the lower validation RMSE.

### Part C: Classification

Built classification models to predict shipment delays.

Selected configuration:

- Random Forest Classifier.
- Seven selected features.
- Prediction threshold: 0.5.

Final test results:

| Metric | Score |
|---|---:|
| Accuracy | 0.867 |
| Precision | 0.689 |
| Recall | 0.756 |
| F1-score | 0.721 |

Confusion matrix:

| | Predicted not delayed | Predicted delayed |
|---|---:|---:|
| Actual not delayed | 125 | 14 |
| Actual delayed | 10 | 31 |

### Part D: Hub Segmentation

Used K-Means clustering to group 24 hubs into three operational segments:

- Standard: 7 hubs.
- Watch: 8 hubs.
- Escalate: 9 hubs.

The segments help identify hubs that may need operational attention.

### Part E: Model Packaging

Saved the trained classification model and supporting files using Joblib and CSV.

### Part F: New Shipment Prediction

Predicted delays for 48 new shipments.

Results:

| Prediction or action | Shipments |
|---|---:|
| Predicted delayed | 8 |
| Predicted not delayed | 40 |
| Escalate to hub manager | 6 |
| Proactive customer call | 2 |
| No action | 40 |

## 6. Repository Structure

```text
data/
    hubs.csv
    raw_shipments.csv
    new_shipments.csv
    modelling_table.csv
    hub_profile.csv

src/
    shipment_prep.py

models/
    delay_model.pkl
    delay_threshold.csv
    feature_order.csv
    impute_values.csv

outputs/
    cluster_separation.csv
    hub_segments.csv
    segment_profile.csv
    new_shipment_predictions.csv

generate_data.py
requirements.txt
README.md
