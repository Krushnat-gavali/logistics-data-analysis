"""Customer segmentation using NumPy K-Means (no scikit-learn required)."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parent))
from data_cleaning import load_data, clean_data

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "outputs" / "customer_segments.csv"

def build_customer_features(df):
    delivered = df["days_to_deliver"].where(df["status"].eq("delivered"))
    d = df.copy()
    d["delivered_days"] = delivered
    features = d.groupby("customer_id").agg(
        order_count=("order_id", "count"),
        total_spend=("order_total", "sum"),
        avg_order_value=("order_total", "mean"),
        total_quantity=("quantity", "sum"),
        avg_delivery_days=("delivered_days", "mean"),
        avg_shipping_cost=("shipping_cost", "mean"),
        cancelled_rate=("status", lambda s: (s == "cancelled").mean()),
        refund_rate=("status", lambda s: (s == "refunded").mean()),
    ).reset_index()
    features["avg_delivery_days"] = features["avg_delivery_days"].fillna(
        features["avg_delivery_days"].median()
    )
    return features

def kmeans_numpy(X, k=4, iterations=100, seed=42):
    rng = np.random.default_rng(seed)
    centroids = X[rng.choice(len(X), size=k, replace=False)].copy()
    labels = np.zeros(len(X), dtype=int)
    for _ in range(iterations):
        distances = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        new_centroids = np.array([
            X[new_labels == j].mean(axis=0) if np.any(new_labels == j)
            else centroids[j]
            for j in range(k)
        ])
        if np.array_equal(new_labels, labels) and np.allclose(new_centroids, centroids):
            labels = new_labels
            centroids = new_centroids
            break
        labels, centroids = new_labels, new_centroids
    return labels, centroids

if __name__ == "__main__":
    df = clean_data(load_data())
    features = build_customer_features(df)
    cols = [
        "order_count", "total_spend", "avg_order_value", "total_quantity",
        "avg_delivery_days", "avg_shipping_cost", "cancelled_rate", "refund_rate"
    ]
    X = features[cols].to_numpy(dtype=float)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1
    Xs = (X - mean) / std

    labels, _ = kmeans_numpy(Xs, k=4)
    features["cluster"] = labels
    RESULT_PATH.parent.mkdir(exist_ok=True)
    features.to_csv(RESULT_PATH, index=False)

    summary = features.groupby("cluster")[cols].mean().round(2)
    print("\nCUSTOMER SEGMENT SUMMARY")
    print(summary)
    print(f"\nSaved: {RESULT_PATH}")
