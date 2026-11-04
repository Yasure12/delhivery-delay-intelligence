
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import clone

# Load modelling data
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
y = df["delivery_hours"]

# Split: 70% training, 15% validation, 15% testing
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

# Compare regression models
models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=6, random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, random_state=42
    )
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    rmse = mean_squared_error(y_val, predictions) ** 0.5
    r2 = r2_score(y_val, predictions)

    results.append({
        "Model": name,
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4)
    })

results_df = pd.DataFrame(results).sort_values("RMSE")

print("VALIDATION RESULTS")
print(results_df.to_string(index=False))

# Select model with lowest validation RMSE
best_model_name = results_df.iloc[0]["Model"]
best_model = clone(models[best_model_name])

# Train on training + validation data
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

best_model.fit(X_train_val, y_train_val)

# Final test evaluation
test_predictions = best_model.predict(X_test)

test_rmse = mean_squared_error(y_test, test_predictions) ** 0.5
test_r2 = r2_score(y_test, test_predictions)

print("\nSELECTED MODEL:", best_model_name)
print("Test RMSE:", round(test_rmse, 4))
print("Test R2:", round(test_r2, 4))

# Cross-validation comparison of the two leading validation models
top_two = results_df.head(2)["Model"].tolist()

cv = KFold(n_splits=5, shuffle=True, random_state=42)

print("\nCROSS-VALIDATION RESULTS")

for name in top_two:
    scores = cross_val_score(
        models[name],
        X_train_val,
        y_train_val,
        cv=cv,
        scoring="r2"
    )

    print(
        name,
        "Scores:", scores.round(3),
        "Mean:", round(scores.mean(), 3),
        "Std:", round(scores.std(), 3)
    )
