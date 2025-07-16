#!/usr/bin/env python3
"""
classify_top5_products.py

Train binary classifiers for the top 5 most‐frequent products,
evaluate each, and plot ROC curves and an AUC bar chart.
"""

import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, classification_report

# ─── CONFIG ─────────────────────────────────────────────────────────
# Absolute paths to your CSVs
ORDERS_CSV = "data/final/orders_with_coords_final_clean.csv"
ITEMS_CSV  = "data/final/order_items_imputed.csv"
TEST_SIZE  = 0.2
RANDOM_SEED = 42

# ─── LOAD DATA & PREP FEATURES ──────────────────────────────────────
orders = pd.read_csv(ORDERS_CSV, parse_dates=["order_date"])
items  = pd.read_csv(ITEMS_CSV)

# Add simple temporal and basket‐size features
orders["dow"]   = orders["order_date"].dt.dayofweek
orders["month"] = orders["order_date"].dt.month
basket_size = items.groupby("order_id")["product_id"].nunique().rename("basket_size")
orders = orders.merge(basket_size, on="order_id", how="left").fillna({"basket_size": 0})

# ─── IDENTIFY TOP 5 PRODUCTS ────────────────────────────────────────
top5 = items["product_id"].value_counts().head(5).index.tolist()
print(f"Top 5 products by frequency: {top5}")

# Prepare ROC plot
plt.figure(figsize=(8, 6))
plt.title("ROC Curves for Top 5 SKU Classifiers")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

auc_scores = []
sku_labels = []

for sku in top5:
    # Create binary label: did this order include the sku?
    orders["bought"] = orders["order_id"].isin(
        items.loc[items.product_id == sku, "order_id"]
    ).astype(int)

    X = orders[["dow", "month", "basket_size"]]
    y = orders["bought"]

    # Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )

    # Train classifier
    model = XGBClassifier(
        n_estimators=100,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=RANDOM_SEED
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred  = model.predict(X_val)
    auc = roc_auc_score(y_val, y_proba)
    acc = accuracy_score(y_val, y_pred)
    print(f"\nSKU {sku}: AUC={auc:.3f}, Accuracy={acc:.3f}")
    print(classification_report(y_val, y_pred, digits=4))

    # Plot ROC
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    plt.plot(fpr, tpr, label=f"{sku} (AUC={auc:.2f})")

    auc_scores.append(auc)
    sku_labels.append(str(sku))

# Plot baseline and legend
plt.plot([0, 1], [0, 1], 'k--', alpha=0.7)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# ─── PLOT AUC BAR CHART ─────────────────────────────────────────────
plt.figure(figsize=(6, 4))
plt.bar(sku_labels, auc_scores, color='skyblue')
plt.title("Validation AUC for Top 5 SKUs")
plt.xlabel("Product ID")
plt.ylabel("AUC")
plt.ylim(0.0, 1.0)
plt.tight_layout()
plt.show()
