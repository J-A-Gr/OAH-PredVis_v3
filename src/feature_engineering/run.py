import pandas as pd

# 1) LOAD RAW DATA
df = pd.read_csv('data/master_transactions.csv')

# 2) CLEAN & DERIVE NET UNITS + REVENUE
df['net_amount'] = df['Amount'] - df['AmountCancelled']
df = df[df['net_amount'] > 0]            # only actual sold units
df['order_date'] = pd.to_datetime(df['CreatedOn_y']).dt.date
df['revenue'] = df['net_amount'] * df['Price_x']

# 3A) BASKET-LEVEL VIEW (one “basket” per order for association rules)
basket_df = (
    df
    .groupby('order_id')
    .agg(
        order_date   = ('order_date',    'first'),
        customer_id  = ('customer_id',   'first'),
        products     = ('product_id',    list)   # list of items in each basket
    )
    .reset_index()
)

# 3B) TIME-SERIES VIEW (daily per SKU for forecasting)
ts_df = (
    df
    .groupby(['order_date', 'product_id'])
    .agg(
        net_units = ('net_amount', 'sum'),
        revenue   = ('revenue',    'sum')
    )
    .reset_index()
)

# 4) QUICK SANITY PRINT
print("Basket-level sample:")
print(basket_df.head(5).to_string(index=False))

print("\nTime-series sample:")
print(ts_df.head(5).to_string(index=False))


basket_df.to_csv('data/work_material/basket_df.csv', index=False)
ts_df.to_csv('data/work_material/ts_df.csv', index=False)
