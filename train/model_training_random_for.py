import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import joblib

# ─── 1. Load product-level data ───────────────────────────────────────────────
products = pd.read_csv(
    'data/final/products_imputed.csv'
)
# Ensure product_id is the same dtype across tables:
products['product_id'] = products['product_id'].astype(int)

# ─── 2. Compute first-sale date per product ───────────────────────────────────
orders = pd.read_csv(
    'data/final/orders_with_coords_final_clean.csv',
    parse_dates=['order_date']
)
order_items = pd.read_csv(
    'data/final/order_items_imputed.csv'
)[['order_id', 'product_id']]
order_items['product_id'] = order_items['product_id'].astype(int)

# join to get order_date on each item
oi = order_items.merge(
    orders[['order_id','order_date']], on='order_id', how='left'
)

# first sale date for each product
first_sale = (
    oi.groupby('product_id')['order_date']
      .min()
      .reset_index()
      .rename(columns={'order_date':'first_sale_date'})
)

# ─── 3. Merge date into products ───────────────────────────────────────────────
df = products.merge(first_sale, on='product_id', how='left')

# if a product was never sold, we can either drop or assign a far-future date:
df['first_sale_date'] = df['first_sale_date'].fillna(pd.Timestamp('2100-01-01'))

# ─── 4. Time‐split at 2025-01-01 ──────────────────────────────────────────────
train_df = df[df['first_sale_date'] < '2025-01-01']
test_df  = df[df['first_sale_date'] >= '2025-01-01']

print(f"Training on {len(train_df)} products (first sold before 2025)")
print(f"Testing on  {len(test_df)} products (first sold in 2025 or never sold)")

# ─── 5. Binarize target ────────────────────────────────────────────────────────
train_df['to_buy_bin'] = (train_df['to_buy'] > 0).astype(int)
test_df['to_buy_bin']  = (test_df['to_buy']  > 0).astype(int)

# ─── 6. Prepare X/y ───────────────────────────────────────────────────────────
def make_X_y(df_):
    # drop non-features:
    X_ = df_.drop(columns=[
        'product_id', 'to_buy', 'to_buy_bin', 'first_sale_date'
    ])
    # keep only numeric:
    X_ = X_.select_dtypes(include=[int, float])
    y_ = df_['to_buy_bin']
    return X_, y_

X_train, y_train = make_X_y(train_df)
X_test,  y_test  = make_X_y(test_df)

# ─── 7. Train Random Forest ───────────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# ─── 8. Evaluate ──────────────────────────────────────────────────────────────
y_proba = rf.predict_proba(X_test)[:, 1]
y_pred  = rf.predict(X_test)

acc    = accuracy_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
precision, recall, _ = precision_recall_curve(y_test, y_proba)
pr_auc  = average_precision_score(y_test, y_proba)

print(f"Test Accuracy: {acc:.3f}")
print(f"Test ROC AUC:  {roc_auc:.3f}")
print(f"Test PR AUC:   {pr_auc:.3f}")

# ─── 9. Plot ROC & PR curves ───────────────────────────────────────────────────
plt.figure()
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC (2025+ Products)')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.2f}")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Random Forest Precision–Recall (2025+ Products)')
plt.legend()
plt.tight_layout()
plt.show()

# ─── 10. Save model ───────────────────────────────────────────────────────────
joblib.dump(rf, 'models/rf_product_time_split.pkl')
