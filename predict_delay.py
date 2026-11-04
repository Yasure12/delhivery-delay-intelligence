
import os
import joblib
import numpy as np
import pandas as pd

os.makedirs("outputs", exist_ok=True)

# Load new shipments and reference data
new_data = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# Clean new shipment data
new_clean = new_data.copy()

new_clean["service_type"] = (
    new_clean["service_type"].astype(str).str.strip().str.title()
)

new_clean["payment_mode"] = new_clean["payment_mode"].fillna("Unknown")

new_clean["declared_value_inr"] = pd.to_numeric(
    new_clean["declared_value"].astype(str)
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False),
    errors="coerce"
)

new_clean = new_clean.drop(columns=["declared_value"])

for col in ["distance_km", "parcel_weight_kg", "declared_value_inr"]:
    new_clean[col] = pd.to_numeric(new_clean[col], errors="coerce")

new_clean.loc[
    new_clean["parcel_weight_kg"] > 100, "parcel_weight_kg"
] = np.nan

# Use saved historical medians
saved_medians = pd.read_csv("models/impute_values.csv")

for _, row in saved_medians.iterrows():
    col = row["column"]
    new_clean[col] = new_clean[col].fillna(row["median_value"])

new_clean["dispatch_time"] = pd.to_datetime(new_clean["dispatch_time"])

# Add hub information
new_model = new_clean.merge(hubs, on="hub_id", how="left")

# Create model features
promised_hours = {
    "Express": 36,
    "Standard": 60,
    "Economy": 96
}

new_model["promised_hours"] = new_model["service_type"].map(promised_hours)

new_model["is_cod"] = (
    new_model["payment_mode"].astype(str).str.lower().str.strip() == "cod"
).astype(int)

new_model["is_weekend"] = (
    new_model["dispatch_time"].dt.dayofweek >= 5
).astype(int)

new_model["dispatch_hour"] = new_model["dispatch_time"].dt.hour

new_model["log_declared_value"] = np.log1p(
    new_model["declared_value_inr"]
)

# Load saved feature order and model
feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()

X_new = new_model[feature_order]

loaded_model = joblib.load("models/delay_model.pkl")

# Predict delay probabilities
probabilities = loaded_model.predict_proba(X_new)[:, 1]

# Load selected threshold
threshold_table = pd.read_csv("models/delay_threshold.csv")
eligible = threshold_table[threshold_table["Recall"] >= 0.80]
threshold = eligible["Threshold"].max()

new_model["delay_probability"] = np.round(probabilities, 4)
new_model["predicted_delay"] = (
    probabilities >= threshold
).astype(int)

# Load hub segments
hub_segments = pd.read_csv("outputs/hub_segments.csv")

new_model = new_model.merge(
    hub_segments[["hub_id", "segment"]],
    on="hub_id",
    how="left"
)

# Recommend an action
def recommend_action(row):
    if row["predicted_delay"] == 0:
        return "No action"
    if row["segment"] == "Escalate":
        return "Escalate to hub manager"
    return "Proactive customer call"

new_model["recommended_action"] = new_model.apply(
    recommend_action, axis=1
)

# Save final predictions
result = new_model[
    [
        "shipment_id",
        "hub_id",
        "service_type",
        "delay_probability",
        "predicted_delay",
        "recommended_action"
    ]
]

result.to_csv("outputs/new_shipment_predictions.csv", index=False)

print("NEW SHIPMENT PREDICTIONS")
print("Total shipments:", len(result))
print("Selected threshold:", threshold)

print("\nPrediction counts:")
print(result["predicted_delay"].value_counts().sort_index())

print("\nRecommended actions:")
print(result["recommended_action"].value_counts())

print("\nFirst five predictions:")
print(result.head().to_string(index=False))

print("\nSaved: outputs/new_shipment_predictions.csv")
