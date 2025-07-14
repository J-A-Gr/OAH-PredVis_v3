import pandas as pd
import os

file_paths = {
    "Customers": r"data/cleaned/customers_clean.csv",
    "OrderItems": r"data/cleaned/order_items_clean.csv",
    "Orders": r"data/cleaned/orders_clean.csv",
    "Products": r"data/cleaned/products_clean.csv",
    "Stock": r"data/cleaned/stock_clean.csv"
}

summary_rows = []

for name, path in file_paths.items():
    df = pd.read_csv(path)

    missing_before = df.isna().sum().sum()
    total_cells = df.size

    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        modes = df[col].mode(dropna=True)
        fill_val = modes[0] if not modes.empty else "Unknown"
        df[col] = df[col].fillna(fill_val)

    missing_after = df.isna().sum().sum()

    # Ensure output directory exists
    os.makedirs("data/imputed", exist_ok=True)

    imputed_path = path.replace("data/cleaned/", "data/imputed/").replace("_clean.csv", "_imputed.csv")
    df.to_csv(imputed_path, index=False)

    summary_rows.append({
        "Dataset": name,
        "Missing Before": missing_before,
        "Missing After": missing_after,
        "Total Cells": total_cells,
        "Output File": os.path.basename(imputed_path)
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
