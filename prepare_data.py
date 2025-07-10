import os
import pandas as pd

"""Master dataset creation - merge other datasets into one"""

# 1. Paths
DATA_DIR    = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "master_dataset.csv")

# 2. Load CSVs (parse dates, disable low_memory warnings)
customers = pd.read_csv(
    os.path.join(DATA_DIR, "oahdotnet_live_Customers.csv"),
    parse_dates=["UpdatedOn"],
    low_memory=False
)
orders = pd.read_csv(
    os.path.join(DATA_DIR, "oahdotnet_live_Orders.csv"),
    parse_dates=["CreatedOn"],
    low_memory=False
)
items = pd.read_csv(
    os.path.join(DATA_DIR, "oahdotnet_live_OrdersItem.csv"),
    parse_dates=["CreatedOn"],
    low_memory=False
)
products = pd.read_csv(
    os.path.join(DATA_DIR, "oahdotnet_live_Products.csv"),
    low_memory=False
)

# 3. Drop exact duplicates
customers.drop_duplicates(inplace=True)
orders.drop_duplicates(inplace=True)
items.drop_duplicates(inplace=True)
products.drop_duplicates(inplace=True)

# 4. Merge orders ← customers
#    - keep order ID as 'order_id'
#    - bring in cust_lifetime_orders and cust_lifetime_value
df = orders.merge(
    customers[["Id", "LifetimeOrders", "LifetimeValue"]],
    left_on="IdCustomer",
    right_on="Id",
    how="left",
    suffixes=("", "_cust")
).rename(columns={
    "Id": "order_id",
    "Id_cust": "customer_id",
    "LifetimeOrders": "cust_lifetime_orders",
    "LifetimeValue": "cust_lifetime_value"
})

# 5. Merge items ← products and aggregate to order-level
#    - rename the item’s Price to item_price
items = items.rename(columns={"Price": "item_price"})
product_meta = products[["Id", "Name", "TotalStock"]].rename(columns={
    "Id": "ProductId",
    "Name": "product_name",
    "TotalStock": "current_stock"
})
itm = items.merge(product_meta, on="ProductId", how="left")

order_agg = (
    itm
    .groupby("OrdersId")
    .agg(
        total_items       = ("Amount",    "sum"),
        avg_item_price    = ("item_price","mean"),
        total_stock_value = (
            "Amount",
            lambda x: (itm.loc[x.index, "current_stock"] * x).sum()
        )
    )
    .reset_index()
)

# 6. Attach aggregated item features back to orders DataFrame
df = df.merge(
    order_agg,
    left_on="order_id",
    right_on="OrdersId",
    how="left"
)

# 7. Derive simple date features from orders’ CreatedOn
df["month"]      = df["CreatedOn"].dt.month
df["dayofweek"]  = df["CreatedOn"].dt.dayofweek
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
df["season"]     = df["month"].map(
    lambda m: "spring" if 3 <= m <= 5
              else "summer" if 6 <= m <= 8
              else "autumn" if 9 <= m <= 11
              else "winter"
)

# 8. Save the master dataset
os.makedirs(DATA_DIR, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

# 9. Sanity-check
print(f"Done! Master dataset saved to: {OUTPUT_FILE}")
print(f"Shape: {df.shape}")
print(df.head())
