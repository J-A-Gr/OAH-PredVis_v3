#!/usr/bin/env python

import pandas as pd

def main():
    # 1. Load order items (already contains order_date)
    items = pd.read_csv(
        'data/final/order_items_imputed.csv',
        parse_dates=['order_date']
    )

    # 2. Extract the calendar date (drops any time component)
    items['sale_day'] = items['order_date'].dt.date

    # 3. Count units sold per SKU per day
    daily = (
        items
        .groupby(['product_id', 'sale_day'])
        .size()
        .reset_index(name='units_sold')
    )

    # 4. Compute each SKU’s avg & std of daily sales
    stats = (
        daily
        .groupby('product_id')['units_sold']
        .agg(
            daily_avg='mean',
            daily_std='std'
        )
        .fillna(0)                  # in case a SKU sold zero days
        .reset_index()
    )

    # 5. Save to CSV for your restock logic
    stats.to_csv(
        'data/final/sales_stats.csv',
        index=False
    )
    print(f"Computed stats for {len(stats)} products and wrote to sales_stats.csv")

if __name__ == '__main__':
    main()
