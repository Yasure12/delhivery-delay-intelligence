
import pandas as pd
import numpy as np

PROMISED_HOURS = {
    "Express": 36,
    "Standard": 60,
    "Economy": 96
}

FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no"
]


def clean_shipments(raw):
    """Remove duplicates, clean columns and fill missing values."""

    clean = raw.drop_duplicates().copy()

    clean["service_type"] = (
        clean["service_type"].astype(str).str.strip().str.title()
    )

    clean["payment_mode"] = clean["payment_mode"].fillna("Unknown")

    clean["declared_value_inr"] = pd.to_numeric(
        clean["declared_value"].astype(str)
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False),
        errors="coerce"
    ).astype(float)

    clean = clean.drop(columns=["declared_value"])

    for col in ["distance_km", "parcel_weight_kg"]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce").astype(float)

    clean["dispatch_time"] = pd.to_datetime(clean["dispatch_time"])

    clean.loc[clean["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    medians = {
        "distance_km": clean["distance_km"].median(),
        "parcel_weight_kg": clean["parcel_weight_kg"].median(),
        "declared_value_inr": clean["declared_value_inr"].median()
    }

    for col, median in medians.items():
        clean[col] = clean[col].fillna(median)

    return clean, medians


def build_features(clean, hubs):
    """Create model features and the delay target."""

    model_df = clean.merge(hubs, on="hub_id", how="left")

    model_df["promised_hours"] = (
        model_df["service_type"].map(PROMISED_HOURS)
    )

    model_df["is_cod"] = (
        model_df["payment_mode"].astype(str).str.lower().str.strip() == "cod"
    ).astype(int)

    model_df["is_weekend"] = (
        model_df["dispatch_time"].dt.dayofweek >= 5
    ).astype(int)

    model_df["dispatch_hour"] = model_df["dispatch_time"].dt.hour

    model_df["log_declared_value"] = np.log1p(
        model_df["declared_value_inr"]
    )

    model_df["is_delayed"] = (
        model_df["delivery_hours"] > model_df["promised_hours"]
    ).astype(int)

    return model_df
