
import os
import pandas as pd
from src.shipment_prep import clean_shipments, build_features

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Load data
raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# Clean shipment data
clean, medians = clean_shipments(raw)

# Create modelling features
model_df = build_features(clean, hubs)

# Save modelling table
model_df.to_csv("data/modelling_table.csv", index=False)

# Basic data checks
print("PART A: DATA CLEANING")
print("---------------------")
print("Raw rows:", len(raw))
print("Duplicate rows removed:", len(raw) - len(clean))
print("Cleaned rows:", len(clean))
print("Modelling rows:", len(model_df))
print("Missing values:", model_df.isnull().sum().sum())

print("\nImputation medians:")
print(pd.Series(medians))

# Delay summary
print("\nDELAY SUMMARY")
print("-------------")
print("Delayed shipments:", model_df["is_delayed"].sum())
print("Delay rate:", round(model_df["is_delayed"].mean(), 4))

# Delay by service type
print("\nDELAY BY SERVICE TYPE")
print(
    model_df.groupby("service_type")["is_delayed"]
    .agg(["count", "sum", "mean"])
    .round(4)
)

# Delay by hub tier
print("\nDELAY BY HUB TIER")
print(
    model_df.groupby("hub_tier")["is_delayed"]
    .agg(["count", "sum", "mean"])
    .round(4)
)

print("\nSaved: data/modelling_table.csv")
