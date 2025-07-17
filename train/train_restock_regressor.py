#!/usr/bin/env python
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import matplotlib.pyplot as plt
import joblib

def build_training_table(items_path, products_path):
    print("1) Loading data…")
    items    = pd.read_csv(items_path, parse_dates=['order_date'])
    products = pd.read_csv(products_path)
    print(f"   order_items: {items.shape}, products: {products.shape}")

    print("2) Merging items with products…")
    df = items.merge(products, on='product_id', how='left')
    print(f"   After merge: {df.shape}")

    print("3) Selecting price…")
    if 'price_y' in df.columns:
        df['price'] = df['price_y']
    elif 'price' in df.columns:
        pass
    else:
        raise KeyError("No price column found")
    df = df.drop(columns=[c for c in ('price_x','price_y') if c in df.columns])

    print("4) Injecting dummy inventory info…")
    df['current_stock']  = 100
    df['lead_time_days'] = 7

    print("5) Sorting & indexing by order_date…")
    df = df.sort_values(['product_id','order_date']).set_index('order_date')

    print("6) Computing sold_7d & sold_30d…")
    df['sold_7d'] = (
        df.groupby('product_id', group_keys=False)['amount']
          .apply(lambda x: x.shift().rolling('7d').sum())
    )
    df['sold_30d'] = (
        df.groupby('product_id', group_keys=False)['amount']
          .apply(lambda x: x.shift().rolling('30d').sum())
    )
    df[['sold_7d','sold_30d']] = df[['sold_7d','sold_30d']].fillna(0)

    print("7) Computing next-window label…")
    def next_window_sales(g):
        lt = int(g['lead_time_days'].iat[0])
        return g['amount'].shift(-1).rolling(f"{lt}d").sum()
    df['label'] = df.groupby('product_id', group_keys=False) \
                    .apply(next_window_sales) \
                    .fillna(0)

    print("8) Resetting index & dropping NaNs…")
    df = df.reset_index()
    before = len(df)
    df = df.dropna(subset=['price','sold_7d','sold_30d','label'])
    print(f"   Dropped {before - len(df)} rows → {len(df)} remain")

    print("9) Building X & y…")
    feature_cols = ['price','current_stock','lead_time_days','sold_7d','sold_30d']
    X = df[feature_cols]
    y = df['label']
    print(f"   X shape: {X.shape}, y shape: {y.shape}")

    return X, y

def main():
    items_path    = 'data/final/order_items_imputed.csv'
    products_path = 'data/final/products_imputed.csv'

    X, y = build_training_table(items_path, products_path)

    print("10) Splitting data…")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   {len(X_train)} train rows, {len(X_val)} val rows")

    print("11) Training LightGBM regressor…")
    model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    print("12) Evaluating model…")
    preds = model.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, preds)
    r2  = r2_score(y_val, preds)
    print(f"   RMSE: {rmse:.3f}")
    print(f"   MAE:  {mae:.3f}")
    print(f"   R²:   {r2:.3f}")

    # --- Matplotlib plots ---
    # a) Actual vs. Predicted
    plt.figure()
    plt.scatter(y_val, preds, alpha=0.3)
    plt.plot([y_val.min(), y_val.max()],
             [y_val.min(), y_val.max()],
             'k--', linewidth=1)
    plt.xlabel('Actual 7-day Demand')
    plt.ylabel('Predicted 7-day Demand')
    plt.title('Actual vs. Predicted Demand')
    plt.tight_layout()
    plt.show()

    # b) Residuals distribution
    residuals = y_val - preds
    plt.figure()
    plt.hist(residuals, bins=50)
    plt.xlabel('Residual (Actual – Predicted)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Residuals')
    plt.tight_layout()
    plt.show()

    print("13) Saving model…")
    model_path = 'models/restock_regressor.pkl'
    joblib.dump(model, model_path)
    print(f"   Model saved to {model_path}")

if __name__ == '__main__':
    main()
