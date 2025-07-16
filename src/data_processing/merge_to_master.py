import os
import pandas as pd
import holidays
import datetime

SRC_DIR = "data/imputed"

# 1. Load and rename key tables
customers = pd.read_csv(os.path.join(SRC_DIR, 'customers_imputed.csv'))
customers.rename(columns={'Id': 'customer_id'}, inplace=True)

orders = pd.read_csv(os.path.join(SRC_DIR, 'orders_imputed.csv'))
orders.rename(columns={
    'Id': 'order_id',
    'IdCustomer': 'customer_id',
    'Status': 'status'
}, inplace=True)
orders['status'] = orders['status'].str.lower()
orders = orders[orders['status'] != 'cancelled']

order_items = pd.read_csv(os.path.join(SRC_DIR, 'order_items_imputed.csv'))
order_items.rename(columns={
    'OrdersId': 'order_id',
    'ProductId': 'product_id'
}, inplace=True)

products = pd.read_csv(os.path.join(SRC_DIR, 'products_imputed.csv'))
products.rename(columns={
    'Id': 'product_id',
    'ProductCode': 'SKU'
}, inplace=True)

stock = pd.read_csv(os.path.join(SRC_DIR, 'stock_imputed.csv'))
stock.rename(columns={'ProductsId': 'product_id'}, inplace=True)

# 2. Merge into one transaction‐level master
master = (
    order_items
    .merge(orders,    on='order_id',    how='inner')
    .merge(products,  on='product_id',  how='left')
    .merge(customers, on='customer_id', how='left')
    .merge(stock,     on='product_id',  how='left')
)

# 3. Merge in geocoded coords
coords = pd.read_csv('data/weather_api/region_coords_geocoded_updated.csv')
master = master.merge(
    coords[['DeliveryCity','DeliveryCountry','lat','lon']],
    on=['DeliveryCity','DeliveryCountry'],
    how='left'
)

missing = (
    master
    .loc[master['lat'].isna(), ['DeliveryCity','DeliveryCountry']]
    .drop_duplicates()
)
if not missing.empty:
    print("⚠️  Missing coordinates for these city–country combos:\n", missing)

# 4. Add holiday features (Belgium, France, Netherlands, Germany, Luxembourg, Lithuania)
#    normalize your order date and build a holiday‐table per country
master['order_date'] = pd.to_datetime(master['CreatedOn_y']).dt.normalize()

# Map full country names to ISO2 codes if needed
country_map = {
    'Belgium': 'BE',
    'Netherlands': 'NL',
    'France': 'FR',
    'Germany': 'DE',
    'Luxembourg': 'LU',
    'Lithuania': 'LT'
}
master['country_code'] = master['DeliveryCountry'].map(country_map).fillna(master['DeliveryCountry'])

def build_holidays_table(country_codes, start_year, end_year):
    """
    Returns DataFrame with columns ['date','country_code','holiday_name'].
    """
    records = []
    years = list(range(start_year, end_year + 1))
    for cc in country_codes:
        cal = holidays.CountryHoliday(cc, years=years)
        for dt, name in cal.items():
            records.append({
                'date': pd.Timestamp(dt),
                'country_code': cc,
                'holiday_name': name
            })
    return pd.DataFrame(records).drop_duplicates(['date','country_code'])

# determine the span of years in your data
min_year = master['order_date'].dt.year.min()
max_year = master['order_date'].dt.year.max()

hols_df = build_holidays_table(
    country_codes=['BE','NL','FR','DE','LU','LT'],
    start_year=min_year,
    end_year=max_year
)

# merge in holidays
master = (
    master
    .merge(
        hols_df,
        left_on=['order_date','country_code'],
        right_on=['date','country_code'],
        how='left'
    )
    .assign(is_holiday=lambda df: df['holiday_name'].notna())
    .drop(columns=['date'])
)

# 5. Save the result
output_path = 'data/master_transactions.csv'
master.to_csv(output_path, index=False)
print(f"✅ Saved {output_path} with {master.shape[0]} rows × {master.shape[1]} cols")
