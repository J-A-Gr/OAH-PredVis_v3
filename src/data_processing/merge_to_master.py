import pandas as pd

# 1. Load and rename key columns
customers = pd.read_csv('data/cleaned/customers_imputed.csv')
customers.rename(columns={'Id': 'customer_id'}, inplace=True)

orders = pd.read_csv('data/cleaned/orders_imputed.csv')
orders.rename(columns={
    'Id': 'order_id',
    'IdCustomer': 'customer_id'
}, inplace=True)

order_items = pd.read_csv('data/cleaned/order_items_imputed.csv')
order_items.rename(columns={
    'OrdersId': 'order_id',
    'ProductId': 'product_id'
}, inplace=True)

products = pd.read_csv('data/cleaned/products_imputed.csv')
products.rename(columns={'Id': 'product_id'}, inplace=True)

stock = pd.read_csv('data/cleaned/stock_imputed.csv')
stock.rename(columns={'ProductsId': 'product_id'}, inplace=True)

# 2. Merge into one transaction-level master
master = (
    order_items
    .merge(orders,      on='order_id',    how='inner')
    .merge(products,    on='product_id',  how='left')
    .merge(customers,   on='customer_id', how='left')
    .merge(stock,       on='product_id',  how='left')
)

# 3. Save the result
master.to_csv('data/master_transactions.csv', index=False)
print(f"✅ Saved data/master_transactions.csv with {master.shape[0]} rows × {master.shape[1]} cols")
