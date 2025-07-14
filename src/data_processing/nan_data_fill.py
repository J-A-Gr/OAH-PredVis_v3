import pandas as pd
import os

# Map of dataset names to filepaths
file_paths = {
    "Customers": "data\cleaned\customers_clean.csv",
    "OrderItems": "data\cleaned\order_items_clean.csv",
    "Orders": "data\cleaned\orders_clean.csv",
    "Products": "data\cleaned\products_clean.csv",
    "Stock": "data\cleaned\stock_clean.csv"
}

summary_rows = []

for name, path in file_paths.items():
    # 1. Load
    df = pd.read_csv(path)
    
    # 2. Count missing before
    missing_before = df.isna().sum().sum()
    total_cells = df.size
    
    # 3. Numeric imputation (median), without chained assignment
    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    
    # 4. Categorical imputation (mode → most frequent, else "Unknown"), without chained assignment
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        modes = df[col].mode(dropna=True)
        fill_val = modes[0] if not modes.empty else "Unknown"
        df[col] = df[col].fillna(fill_val)
    
    # 5. Count missing after
    missing_after = df.isna().sum().sum()
    
    # 6. Save imputed version
    imputed_path = path.replace("_clean.csv", "_imputed.csv")
    df.to_csv(imputed_path, index=False)
    
    # 7. Record summary
    summary_rows.append({
        "Dataset": name,
        "Missing Before": missing_before,
        "Missing After": missing_after,
        "Total Cells": total_cells,
        "Output File": os.path.basename(imputed_path)
    })

# 8. Print summary
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))