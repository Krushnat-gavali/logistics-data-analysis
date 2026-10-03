# Logistics Data Analysis

A beginner-friendly logistics data analysis project for **Week 1: Strategic Planning and Data Exploration in Logistics**.

## Dataset

This version uses the **Sample E-commerce Orders Dataset** from Cotera.

Source: https://cotera.co/datasets/library/sample-ecommerce-orders

The source page describes the dataset as synthetic e-commerce order data and lists it under **CC0 1.0**. The CSV is kept outside GitHub and is used locally for analysis.

## Objective

The project studies:
- order and delivery performance
- delivery completion
- delivery time
- shipping cost
- order value
- cancellations and refunds
- product/category patterns
- customer segmentation
- basic delivery-time prediction

## Dataset size

- 50,000 orders
- 20 columns
- 12,854 unique customers
- 6 product categories
- 8 countries

## Main KPIs from the supplied dataset

- Total orders: 50,000
- Delivered orders: 35,579
- Delivery completion rate: 71.16%
- Average delivery time for delivered orders: 6.00 days
- Median delivery time: 6 days
- Average shipping cost: 5.67
- Average order value: 187.20
- Cancellation rate: 5.12%
- Refund rate: 4.01%
- Average quantity per order: 1.59

## Methods

The project uses **Python, Pandas, NumPy and Matplotlib**.

- KPI analysis uses Pandas.
- EDA uses Pandas and Matplotlib.
- Delivery-time prediction uses NumPy linear regression.
- Customer segmentation uses a NumPy K-Means implementation.
- Scikit-learn is not required.

## Folder structure

```text
logistics-data-analysis/
├── data/
│   └── README.md
├── src/
│   ├── data_cleaning.py
│   ├── kpi_analysis.py
│   ├── eda.py
│   ├── prediction.py
│   └── clustering.py
├── outputs/
├── requirements.txt
├── .gitignore
└── README.md
```

## How to run

1. Put `sample-ecommerce-orders.csv` inside `data/`.
2. Open the project in VS Code.
3. In the project terminal run:

```powershell
py src/data_cleaning.py
py src/kpi_analysis.py
py src/eda.py
py src/prediction.py
py src/clustering.py
```

The generated CSV results and charts are saved in `outputs/`.

## Privacy and repository practice

The dataset contains a `customer_email` field. It is excluded from modelling and is not uploaded to GitHub. The raw CSV is also ignored through `.gitignore`.
