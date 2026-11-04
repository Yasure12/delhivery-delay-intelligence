
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

os.makedirs("models", exist_ok=True)

# Load data
df = pd.read_csv("data/modelling_table.csv")

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg",
    "log_declared_value", "n_stops", "rain_flag",
    "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity", "label_print_batch",
    "scanner_build_no", "tape_roll_no", "sticker_serial_mod",
    "shelf_slot_no"
]

X = df[FEATURES]
y = df["is_delayed"]

# Stratified train, validation and test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("DATA SPLIT")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))

# Compare feature counts
print("\nFEATURE SELECTION")

for k in [5, 7, 10, 12, 17]:
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_k = selector.fit_transform(X_train, y_train)
    X_val_k = selector.transform(X_val)

    model = RandomForestClassifier(
        n_estimators=100, random_state=42
    )
    model.fit(X_train_k, y_train)
    predictions = model.predict(X_val_k)

    print(
        "Features:", k,
        "Validation F1:", round(f1_score(y_val, predictions), 3)
    )

# Pipeline for tuning
pipeline = Pipeline([
    ("selector", SelectKBest(score_func=f_classif, k=7)),
    ("model", RandomForestClassifier(random_state=42))
])

param_grid = {
    "model__n_estimators": [50, 100, 150],
    "model__max_depth": [4, 6, 8],
    "model__min_samples_leaf": [1, 5, 10]
}

# Grid search
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=1
)
grid_search.fit(X_train, y_train)

print("\nGRID SEARCH")
print("Best parameters:", grid_search.best_params_)
print("Best CV accuracy:", round(grid_search.best_score_, 3))

# Random search
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=9,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=1
)
random_search.fit(X_train, y_train)

print("\nRANDOM SEARCH")
print("Best parameters:", random_search.best_params_)
print("Best CV accuracy:", round(random_search.best_score_, 3))

# Compare validation models
default_model = Pipeline([
    ("selector", SelectKBest(score_func=f_classif, k=7)),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])

candidates = {
    "Default RF": default_model,
    "Random Search RF": random_search.best_estimator_,
    "Grid Search RF": grid_search.best_estimator_
}

validation_results = []

for name, model in candidates.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    validation_results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_val, predictions), 3),
        "F1": round(f1_score(y_val, predictions), 3)
    })

validation_df = pd.DataFrame(validation_results)

print("\nVALIDATION MODEL COMPARISON")
print(validation_df.to_string(index=False))

# Select default RF based on the validation comparison
final_classifier = default_model

# Fit final classifier on training data
final_classifier.fit(X_train, y_train)

# Choose threshold using validation recall
validation_probabilities = final_classifier.predict_proba(X_val)[:, 1]

threshold_results = []

for threshold in np.arange(0.1, 1.0, 0.1):
    predictions = (validation_probabilities >= threshold).astype(int)

    threshold_results.append({
        "Threshold": round(threshold, 1),
        "Precision": round(precision_score(y_val, predictions, zero_division=0), 3),
        "Recall": round(recall_score(y_val, predictions, zero_division=0), 3),
        "F1": round(f1_score(y_val, predictions, zero_division=0), 3)
    })

threshold_df = pd.DataFrame(threshold_results)

print("\nTHRESHOLD COMPARISON")
print(threshold_df.to_string(index=False))

# Select highest threshold with validation recall >= 0.80
eligible = threshold_df[threshold_df["Recall"] >= 0.80]
threshold = eligible["Threshold"].max()

threshold_df.to_csv("models/delay_threshold.csv", index=False)

print("\nSelected threshold:", threshold)

# Final test evaluation
test_probabilities = final_classifier.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= threshold).astype(int)

print("\nFINAL TEST RESULTS")
print("Accuracy:", round(accuracy_score(y_test, test_predictions), 3))
print("Precision:", round(precision_score(y_test, test_predictions), 3))
print("Recall:", round(recall_score(y_test, test_predictions), 3))
print("F1:", round(f1_score(y_test, test_predictions), 3))
print("Confusion matrix:")
print(confusion_matrix(y_test, test_predictions))
