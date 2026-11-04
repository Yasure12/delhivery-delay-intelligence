
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from itertools import combinations

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Load modelling data
df = pd.read_csv("data/modelling_table.csv")

# Calculate hub performance profile
hub_profile = (
    df.groupby("hub_id")
    .agg(
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean"),
        shipment_count=("shipment_id", "count")
    )
    .reset_index()
)

hub_profile.to_csv("data/hub_profile.csv", index=False)

# Features used for clustering
cluster_features = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops"
]

X = hub_profile[cluster_features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compare different values of K
separation_results = []

for k in [2, 3, 4, 5]:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    sizes = pd.Series(labels).value_counts().sort_index().tolist()

    # Average distance of hubs from their assigned centroid
    distances = np.linalg.norm(
        X_scaled - kmeans.cluster_centers_[labels],
        axis=1
    )
    within_distance = distances.mean()

    # Average distance between cluster centroids
    centroid_distances = [
        np.linalg.norm(a - b)
        for a, b in combinations(kmeans.cluster_centers_, 2)
    ]
    between_distance = np.mean(centroid_distances)

    separation_results.append({
        "K": k,
        "Cluster_Sizes": sizes,
        "Smallest_Cluster": min(sizes),
        "Within_Distance": round(within_distance, 4),
        "Between_Distance": round(between_distance, 4),
        "Separation_Ratio": round(
            between_distance / within_distance, 4
        )
    })

separation_df = pd.DataFrame(separation_results)
separation_df.to_csv(
    "outputs/cluster_separation.csv", index=False
)

# Final clustering with K = 3
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

hub_profile["cluster"] = kmeans.fit_predict(X_scaled)

# Give cluster names based on average delay rate
cluster_order = (
    hub_profile.groupby("cluster")["delay_rate"]
    .mean()
    .sort_values()
)

segment_names = {
    cluster_order.index[0]: "Standard",
    cluster_order.index[1]: "Watch",
    cluster_order.index[2]: "Escalate"
}

hub_profile["segment"] = hub_profile["cluster"].map(segment_names)

# Save hub segments
hub_profile.to_csv("outputs/hub_segments.csv", index=False)

# Create segment summary
segment_profile = (
    hub_profile.groupby("segment")
    .agg(
        hub_count=("hub_id", "count"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean")
    )
    .reset_index()
)

segment_profile.to_csv(
    "outputs/segment_profile.csv", index=False
)

# Display results
print("HUB PROFILE")
print(hub_profile.to_string(index=False))

print("\nCLUSTER SEPARATION")
print(separation_df.to_string(index=False))

print("\nSEGMENT PROFILE")
print(segment_profile.round(4).to_string(index=False))

print("\nSaved Part D output files.")
