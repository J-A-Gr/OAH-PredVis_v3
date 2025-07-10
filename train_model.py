import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# 1) Load the master dataset
df = pd.read_csv(
    "data/master_dataset.csv",
    parse_dates=["CreatedOn"],
    low_memory=False
)

# 2) Basic missing-value handling
df.fillna(0, inplace=True)

# 3) One-hot encode categorical features
cat_cols = ["season", "Status", "DeliveryCountry"]
df = pd.get_dummies(df, columns=[c for c in cat_cols if c in df.columns], drop_first=True)

# 4) Time-based train/test split:
cutoff = df["CreatedOn"].max() - pd.DateOffset(years=1)
train_df = df[df["CreatedOn"] < cutoff].copy()
test_df  = df[df["CreatedOn"] >= cutoff].copy()

# 5) Define target and drop-list
target   = "total_items"
drop_cols = {"order_id", "OrdersId", "IdCustomer", "CreatedOn", target} & set(df.columns)

# 6) Select numeric features only
feature_cols = [
    c for c in train_df.columns
    if c not in drop_cols and pd.api.types.is_numeric_dtype(train_df[c])
]

X_train, y_train = train_df[feature_cols], train_df[target]
X_test,  y_test  = test_df[feature_cols],  test_df[target]

print(f"Using {len(feature_cols)} numeric features.")

# 7) Train RandomForest
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
print("Training RandomForest…")
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
mse_rf  = mean_squared_error(y_test, pred_rf)
rmse_rf = np.sqrt(mse_rf)
print(f"RandomForest RMSE: {rmse_rf:.2f}")

# 8) Train XGBoost
xgbr = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42, n_jobs=-1)
print("Training XGBoost…")
xgbr.fit(X_train, y_train)
pred_xgb = xgbr.predict(X_test)
mse_xgb  = mean_squared_error(y_test, pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
print(f"XGBoost RMSE: {rmse_xgb:.2f}")

# 9) Feature importance for RandomForest
importances = pd.Series(rf.feature_importances_, index=feature_cols)
print("\nTop 10 RandomForest feature importances:")
print(importances.nlargest(10))
