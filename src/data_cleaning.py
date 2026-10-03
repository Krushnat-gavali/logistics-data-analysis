"""Load and prepare the sample e-commerce orders dataset."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sample-ecommerce-orders.csv"
OUTPUT_PATH = ROOT / "outputs" / "cleaned_orders.csv"

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["ordered_at"] = pd.to_datetime(df["ordered_at"], errors="coerce")
    numeric_cols = [
        "quantity", "unit_price", "discount_pct", "shipping_cost",
        "order_total", "refund_amount", "days_to_deliver"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def clean_data(df):
    df = df.drop_duplicates(subset="order_id").copy()
    df = df[df["quantity"] > 0].copy()
    df["discount_pct"] = df["discount_pct"].clip(lower=0, upper=1)
    df["shipping_cost"] = df["shipping_cost"].clip(lower=0)
    df["order_total"] = df["order_total"].clip(lower=0)
    return df

if __name__ == "__main__":
    df = clean_data(load_data())
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Rows after cleaning: {len(df):,}")
    print(f"Saved: {OUTPUT_PATH}")
