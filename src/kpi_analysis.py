"""Calculate logistics KPIs for the sample e-commerce dataset."""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from data_cleaning import load_data, clean_data

def calculate_kpis(df):
    total = len(df)
    delivered = df[df["status"].eq("delivered")]
    return {
        "Total Orders": total,
        "Delivered Orders": len(delivered),
        "Delivery Completion Rate (%)": len(delivered) / total * 100,
        "Average Delivery Time (days)": delivered["days_to_deliver"].mean(),
        "Median Delivery Time (days)": delivered["days_to_deliver"].median(),
        "Average Shipping Cost": df["shipping_cost"].mean(),
        "Average Order Value": df["order_total"].mean(),
        "Cancellation Rate (%)": df["status"].eq("cancelled").mean() * 100,
        "Refund Rate (%)": df["status"].eq("refunded").mean() * 100,
        "Average Quantity per Order": df["quantity"].mean(),
    }

if __name__ == "__main__":
    df = clean_data(load_data())
    kpis = calculate_kpis(df)
    print("\nLOGISTICS KPI SUMMARY")
    print("-" * 40)
    for name, value in kpis.items():
        if isinstance(value, float):
            print(f"{name}: {value:.2f}")
        else:
            print(f"{name}: {value:,}")
