"""Predict delivery time using NumPy linear regression (no scikit-learn required)."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parent))
from data_cleaning import load_data, clean_data

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "outputs" / "prediction_results.csv"

def prepare_data(df):
    d = df[df["status"].eq("delivered")].copy()
    # Avoid customer_email and IDs: they are identifiers, not useful predictive features.
    numeric = [
        "quantity", "unit_price", "discount_pct",
        "shipping_cost", "order_total"
    ]
    categorical = ["country", "channel", "device", "product_category", "payment_method"]
    X = pd.concat(
        [d[numeric], pd.get_dummies(d[categorical], drop_first=True, dtype=float)],
        axis=1
    )
    y = d["days_to_deliver"].to_numpy(dtype=float)
    X = X.to_numpy(dtype=float)
    return X, y

def standardize_fit(X_train):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1
    return mean, std

def fit_ols(X_train, y_train):
    X1 = np.column_stack([np.ones(len(X_train)), X_train])
    beta = np.linalg.pinv(X1.T @ X1) @ X1.T @ y_train
    return beta

def run_prediction(df, test_size=0.2, seed=42):
    X, y = prepare_data(df)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    split = int(len(y) * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    mean, std = standardize_fit(X_train)
    X_train_s = (X_train - mean) / std
    X_test_s = (X_test - mean) / std
    beta = fit_ols(X_train_s, y_train)

    pred = np.column_stack([np.ones(len(X_test_s)), X_test_s]) @ beta
    mae = np.mean(np.abs(y_test - pred))
    ss_res = np.sum((y_test - pred) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
    return mae, r2, y_test, pred

if __name__ == "__main__":
    df = clean_data(load_data())
    mae, r2, actual, predicted = run_prediction(df)
    result = pd.DataFrame({"actual_days": actual, "predicted_days": predicted})
    RESULT_PATH.parent.mkdir(exist_ok=True)
    result.to_csv(RESULT_PATH, index=False)
    print(f"MAE: {mae:.2f} days")
    print(f"R²: {r2:.3f}")
    print(f"Saved predictions: {RESULT_PATH}")
