"""Exploratory data analysis and charts."""
from pathlib import Path
import sys
import matplotlib.pyplot as plt
sys.path.append(str(Path(__file__).resolve().parent))
from data_cleaning import load_data, clean_data

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def save_status_chart(df):
    counts = df["status"].value_counts()
    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.title("Order Status Distribution")
    plt.xlabel("Order Status")
    plt.ylabel("Number of Orders")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "order_status_distribution.png", dpi=150)
    plt.close()

def save_delivery_chart(df):
    delivered = df.loc[df["status"].eq("delivered"), "days_to_deliver"]
    plt.figure(figsize=(8, 5))
    plt.hist(delivered, bins=11)
    plt.title("Delivery Time Distribution")
    plt.xlabel("Days to Deliver")
    plt.ylabel("Number of Delivered Orders")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "delivery_time_distribution.png", dpi=150)
    plt.close()

def save_category_chart(df):
    counts = df["product_category"].value_counts()
    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.title("Orders by Product Category")
    plt.xlabel("Product Category")
    plt.ylabel("Number of Orders")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "orders_by_category.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    df = clean_data(load_data())
    save_status_chart(df)
    save_delivery_chart(df)
    save_category_chart(df)
    print(f"Charts saved to: {FIG_DIR}")
