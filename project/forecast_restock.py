#!/usr/bin/env python3
"""
forecast_restock_with_metrics.py

Loads CSVs, aggregates weekly demand, builds lag-feature ML regression with XGBoost,
evaluates on 2025 data using multiple metrics, then visualizes metrics and predicted vs actual,
and finally forecasts the next N weeks’ demand per SKU for restock planning.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import matplotlib.pyplot as plt

# 1. CONFIG
DATA_DIR       = "data/final"
ORDERS_CSV     = f"{DATA_DIR}/orders_with_coords_final_clean.csv"
ITEMS_CSV      = f"{DATA_DIR}/order_items_imputed.csv"
FORECAST_WEEKS = 4
CUTOFF_DATE    = "2025-01-01"
LAG_WEEKS      = [1, 2, 3, 4, 52]

# 2. LOAD & AGGREGATE
orders = pd.read_csv(ORDERS_CSV, parse_dates=["order_date"])
items  = pd.read_csv(ITEMS_CSV)
df = orders[["order_date","order_id"]].merge(
    items[["order_id","product_id"]], on="order_id"
)
df["week_start"] = df["order_date"].dt.to_period("W-MON").apply(lambda r: r.start_time)
weekly = df.groupby(["product_id","week_start"]).size().reset_index(name="sales")

# 3. FEATURE ENGINEERING
weekly["weekofyear"] = weekly["week_start"].dt.isocalendar().week
weekly["month"]      = weekly["week_start"].dt.month
for lag in LAG_WEEKS:
    weekly[f"lag_{lag}"] = weekly.groupby("product_id")["sales"].shift(lag)
lag_cols = [f"lag_{lag}" for lag in LAG_WEEKS]
weekly = weekly.dropna(subset=lag_cols)

# 4. SPLIT TRAIN / TEST
train = weekly[weekly["week_start"] < CUTOFF_DATE].copy()
test  = weekly[weekly["week_start"] >= CUTOFF_DATE].copy()
train_pids = train["product_id"].unique()
test = test[test["product_id"].isin(train_pids)].copy()

# 5. ENCODE PRODUCT ID
le = LabelEncoder()
train["prod_idx"] = le.fit_transform(train["product_id"])
test["prod_idx"]  = le.transform(test["product_id"])
features = ["prod_idx","weekofyear","month"] + lag_cols
X_train, y_train = train[features], train["sales"]
X_test,  y_test  = test[features],  test["sales"]

# 6. TRAIN & EVALUATE
model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print("Evaluation on 2025 test set:")
print(f"  MAPE: {mape:.2%}")
print(f"   MAE: {mae:.2f} units")
print(f"  RMSE: {rmse:.2f} units")
print(f"    R²: {r2:.3f}")

# 7. VISUALIZE METRICS & PREDICTION
metrics = {"MAPE": mape, "MAE": mae, "RMSE": rmse, "R2": r2}
names, vals = zip(*metrics.items())

plt.figure(figsize=(8,5))
plt.bar(names, vals)
plt.title("Model Evaluation Metrics")
plt.ylabel("Value")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', linewidth=1)
plt.xlabel("Actual Weekly Sales")
plt.ylabel("Predicted Weekly Sales")
plt.title("Predicted vs Actual Sales")
plt.tight_layout()
plt.show()

# 8. FORECAST NEXT WEEKS
last_week = weekly["week_start"].max()
future_weeks = [last_week + pd.Timedelta(weeks=i) for i in range(1, FORECAST_WEEKS+1)]
recent = weekly[weekly["week_start"] == last_week].set_index("product_id")["sales"].to_dict()

future_rows = []
for pid in le.classes_:
    lag_vals = {f"lag_{lag}": recent.get(pid, 0) for lag in LAG_WEEKS}
    for wk in future_weeks:
        future_rows.append({
            "product_id": pid,
            "week_start": wk,
            "weekofyear": wk.isocalendar().week,
            "month": wk.month,
            **lag_vals
        })

future = pd.DataFrame(future_rows)
future["prod_idx"] = le.transform(future["product_id"])
future["forecast"] = model.predict(future[features])

restock = (
    future.groupby("product_id")["forecast"]
          .sum()
          .reset_index(name=f"pred_next_{FORECAST_WEEKS}wks")
          .sort_values(f"pred_next_{FORECAST_WEEKS}wks", ascending=False)
)

print("\nTop products by forecasted demand (next "
      f"{FORECAST_WEEKS} weeks):")
print(restock.head(20).to_string(index=False))
