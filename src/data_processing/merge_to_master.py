import pandas as pd

# 1. Read CSV files
orders = pd.read_csv('data/final/orders_with_coords_final_clean.csv')
order_items = pd.read_csv('data/final/order_items_imputed.csv')
products = pd.read_csv('data/final/products_imputed.csv')
customers = pd.read_csv('data/final/customers_imputed.csv')

# 2. Parse date column in orders (adjust the column name if different)
orders['order_date'] = pd.to_datetime(orders['order_date'])

# 3. Merge datasets: inner joins to keep only matching records
df = order_items.merge(orders, on='order_id', how='inner') \
                .merge(products, on='product_id', how='inner') \
                .merge(customers, on='customer_id', how='inner')

# 4. Save the master dataset to /mnt/data
output_path = 'data/master_dataset.csv'
df.to_csv(output_path, index=False)

output_path
