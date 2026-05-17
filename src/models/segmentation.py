"""
src/models/segmentation.py
Customer segmentation using RFM feature engineering + K-Means clustering.
Assigns human-readable segment labels based on cluster centroids.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


SEGMENT_NAMES = {
    0: "Champions",
    1: "Loyal Customers",
    2: "At-Risk",
    3: "New Customers",
}

SEGMENT_COLORS = {
    "Champions":       "#378ADD",
    "Loyal Customers": "#1D9E75",
    "At-Risk":         "#BA7517",
    "New Customers":   "#7F77DD",
}

SEGMENT_ACTIONS = {
    "Champions":       "Reward with exclusive perks. Activate as brand ambassadors.",
    "Loyal Customers": "Upsell premium tiers. Offer subscription bundles.",
    "At-Risk":         "Trigger win-back email sequence. Offer targeted discount.",
    "New Customers":   "Optimize onboarding. Drive second purchase within 30 days.",
}


def fit_kmeans(rfm: pd.DataFrame, n_clusters: int = 4, random_state: int = 42) -> pd.DataFrame:
    """
    Fits K-Means on scaled RFM features.
    Returns rfm DataFrame with added 'cluster', 'segment', 'color' columns.
    """
    features = rfm[["recency", "frequency", "monetary"]].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    rfm = rfm.copy()
    rfm["cluster"] = km.fit_predict(X_scaled)

    # Map cluster IDs to meaningful labels based on centroid ordering
    centroids   = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_),
                               columns=["recency", "frequency", "monetary"])
    centroids["cluster"] = range(n_clusters)

    # Rank: Champions = low recency (recent), high freq & monetary
    centroids["rank"] = (
        -centroids["recency"]        # lower recency = more recent = better
        + centroids["frequency"] * 50
        + centroids["monetary"] * 0.1
    )
    rank_order = centroids.sort_values("rank", ascending=False)["cluster"].tolist()

    label_map = {cluster_id: SEGMENT_NAMES[rank] for rank, cluster_id in enumerate(rank_order)}
    rfm["segment"] = rfm["cluster"].map(label_map)
    rfm["color"]   = rfm["segment"].map(SEGMENT_COLORS)
    rfm["action"]  = rfm["segment"].map(SEGMENT_ACTIONS)

    sil = silhouette_score(X_scaled, rfm["cluster"])
    print(f"[Segmentation] Silhouette Score: {sil:.3f}")

    return rfm


def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """Returns per-segment aggregate statistics for display."""
    summary = (
        rfm.groupby("segment")
           .agg(
               users=("user_id",   "count"),
               avg_recency=("recency",   "mean"),
               avg_frequency=("frequency", "mean"),
               avg_monetary=("monetary",  "mean"),
               total_revenue=("monetary",  "sum"),
           )
           .reset_index()
    )
    for col in ["avg_recency", "avg_frequency", "avg_monetary", "total_revenue"]:
        summary[col] = summary[col].round(1)
    summary["color"]  = summary["segment"].map(SEGMENT_COLORS)
    summary["action"] = summary["segment"].map(SEGMENT_ACTIONS)
    return summary
