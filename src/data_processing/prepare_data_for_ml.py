import pandas as pd

def load_and_clean(path):
    print("Available columns:", pd.read_csv(path, nrows=0).columns.tolist())

    date_col   = 'CreatedOn_x'
    region_col = 'DeliveryCountry'
    qty_col    = 'Amount'
    prod_col   = 'Name'

    df = pd.read_csv(path, parse_dates=[date_col])
    df.rename(columns={
        date_col: 'order_date',
        region_col: 'region',
        qty_col:    'quantity',
        prod_col:   'product_name'
    }, inplace=True)

    df.drop_duplicates(inplace=True)
    # avoid chained‐assignment warning
    df['region'] = df['region'].fillna('Unknown')
    return df

def feature_engineering(df):
    df['month']       = df['order_date'].dt.month
    df['day_of_week'] = df['order_date'].dt.dayofweek
    return df

def create_basket(df):
    basket = df.groupby(['order_id','product_name'])['quantity'] \
               .sum() \
               .unstack(fill_value=0)
    # future-proof conversion
    basket = (basket > 0).astype(int)
    return basket

if __name__ == '__main__':
    df = load_and_clean('data/master_transactions.csv')
    df = feature_engineering(df)

    # export time-series table
    df.to_csv('data/work_material/ts_df.csv', index=False)
    # export basket matrix
    basket = create_basket(df)
    basket.to_csv('data/work_material/basket_df.csv')

    print("✅ Data prepared: ts_df.csv and basket_df.csv")
