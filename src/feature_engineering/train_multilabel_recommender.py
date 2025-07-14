import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib
import matplotlib.pyplot as plt

# --- USER ADJUSTABLE ---
MASTER_CSV    = 'data/master_transactions.csv'
BASKET_CSV    = 'data/work_material/basket_df.csv'
MODEL_OUT     = 'models/bundle_recommender_xgb.pkl'
TOP_N_PRODUCT = 50     # how many products to keep as labels
TEST_SIZE     = 0.25    # fraction held out for evaluation
# -----------------------

# 1) Load and rename
df = pd.read_csv(
    MASTER_CSV,
    parse_dates=['CreatedOn_x']
).rename(columns={
    'CreatedOn_x': 'order_date',
    'Amount': 'quantity',
    'Name': 'product_name',
    'DeliveryCountry': 'region'
})

# 2) Pick top-N products
top_products = (
    df['product_name']
      .value_counts()
      .head(TOP_N_PRODUCT)
      .index
      .tolist()
)

# 3) Build per-order features
order_feat = (
    df.groupby('order_id')
      .agg({
          'order_date': 'first',
          'region':     'first',
          'quantity':   'sum'
      })
      .reset_index()
)
order_feat['month']       = order_feat['order_date'].dt.month
order_feat['day_of_week'] = order_feat['order_date'].dt.dayofweek

# 4) Load basket matrix and subset top products
basket = (
    pd.read_csv(BASKET_CSV, index_col=0)
      [top_products]
      .fillna(0)
      .astype(int)
)

# 5) Align features & labels
data = (
    order_feat
      .set_index('order_id')
      .join(basket, how='inner')
)
X = data[['region','month','day_of_week','quantity']]
y = data[top_products]

# 6) Encode region
X = pd.get_dummies(X, columns=['region'], drop_first=True)

# 7) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42
)

# 8) Fit One-vs-Rest XGBoost
clf = OneVsRestClassifier(
    XGBClassifier(
        eval_metric='logloss',
        n_estimators=100,
        max_depth=4
    )
)
clf.fit(X_train, y_train)

# 9) Predict & evaluate
y_pred = clf.predict(X_test)
report = classification_report(
    y_test, y_pred,
    zero_division=0,
    output_dict=True
)
report_df = pd.DataFrame(report).T

print("\nClassification report for top products:\n")
print(report_df)

# 10) Save model
joblib.dump(clf, MODEL_OUT)
print(f"\n✅ Model saved to {MODEL_OUT}")

# 11) Plot F1-score per product
f1_vals = report_df.iloc[:len(top_products)]['f1-score']
f1_vals.index = top_products

plt.figure(figsize=(10, 6))
plt.bar(f1_vals.index, f1_vals.values)
plt.title('F1-score for Top Products')
plt.xlabel('Product Name')
plt.ylabel('F1-score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('docs/f1_scores_per_product.png')
plt.show()

# 12) Product checker utility
def check_product_stats(product_name, df, basket, y_test=None, y_pred=None, report_df=None):
    print(f"\n🔍 Checking stats for product: '{product_name}'")

    # 1. Total sales count
    total_sales = df['product_name'].value_counts().get(product_name, 0)
    print(f"🛒 Total times sold: {total_sales}")

    # 2. Presence in basket matrix
    in_basket = product_name in basket.columns
    print(f"📦 Present in basket matrix: {in_basket}")

    # 3. If test labels & predictions are given
    if y_test is not None and y_pred is not None:
        if product_name in y_test.columns:
            true_count = y_test[product_name].sum()
            pred_count = pd.DataFrame(y_pred, columns=y_test.columns)[product_name].sum()

            print(f"✅ In test set: Yes — appears in {true_count} test samples")
            print(f"🤖 Predicted as present in: {pred_count} test samples")

            if report_df is not None and product_name in report_df.index:
                row = report_df.loc[product_name]
                print(f"\n📈 Classification Report:")
                print(f"Precision:  {row['precision']:.2f}")
                print(f"Recall:     {row['recall']:.2f}")
                print(f"F1-score:   {row['f1-score']:.2f}")
                print(f"Support:    {int(row['support'])}")
            else:
                print("📉 No classification report available for this product.")
        else:
            print("⚠️ Product is not part of test label set.")
    print("-" * 60)

# 13) Interactive input loop
while True:
    user_input = input("\nEnter part of a product name to check (or 'exit'): ").strip()
    if user_input.lower() == 'exit':
        break

    # Safely process product names
    product_names = df['product_name'].dropna().astype(str).unique()
    matches = [p for p in product_names if user_input.lower() in p.lower()]

    if not matches:
        print("❌ No matching products found.")
        continue

    print(f"\n🔎 Found {len(matches)} match(es):")
    for product in matches:
        check_product_stats(
            product_name=product,
            df=df,
            basket=basket,
            y_test=y_test,
            y_pred=y_pred,
            report_df=report_df
        )
