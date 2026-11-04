
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

from src.shipment_prep import clean_shipments, build_features, FEATURES

os.makedirs("models", exist_ok=True)

# Load and prepare historical data
raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

clean, medians = clean_shipments(raw)
model_df = build_features(clean, hubs)

X = model_df[FEATURES]
y = model_df["is_delayed"]

# Use the same training split as Part C
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Build and train the selected classifier
final_model = Pipeline([
    ("selector", SelectKBest(score_func=f_classif, k=7)),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

final_model.fit(X_train, y_train)

# Save model
joblib.dump(final_model, "models/delay_model.pkl")

# Save feature order
pd.DataFrame({"feature": FEATURES}).to_csv(
    "models/feature_order.csv", index=False
)

# Save historical imputation values
pd.DataFrame({
    "column": list(medians.keys()),
    "median_value": list(medians.values())
}).to_csv("models/impute_values.csv", index=False)

print("MODEL PACKAGING COMPLETED")
print("Model saved: models/delay_model.pkl")
print("Feature order saved: models/feature_order.csv")
print("Imputation values saved: models/impute_values.csv")
print("Training rows:", len(X_train))
print("Features:", len(FEATURES))
