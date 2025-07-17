#!/usr/bin/env python

import pandas as pd
import numpy as np

def main():
    # 1) Load your model predictions + inventory info
    preds = pd.read_csv('data/final/model_preds.csv')
    #   must include: product_id,p_sell,current_stock,lead_time_days

    # 2) Load historical sales stats
    stats = pd.read_csv('data/final/sales_stats.csv')
    #   contains: product_id,daily_avg,daily_std

    # 3) Merge and fill missing stats with 0 (never sold)
    df = preds.merge(stats, on='product_id', how='left')
    df[['daily_avg','daily_std']] = df[['daily_avg','daily_std']].fillna(0)

    # 4) Compute expected demand over lead time
    df['exp_demand_LT'] = df['daily_avg'] * df['lead_time_days']

    # 5) Compute safety stock (95% service level, z ≈ 1.65)
    z = 1.65
    df['safety_stock'] = z * df['daily_std'] * np.sqrt(df['lead_time_days'])

    # 6) Reorder point = expected demand + safety stock
    df['reorder_point'] = df['exp_demand_LT'] + df['safety_stock']

    # 7) Final restock flag: true if on-hand < reorder_point
    df['need_reorder'] = df['current_stock'] < df['reorder_point']

    # 8) Export only the SKUs to reorder
    restock = df.loc[ df['need_reorder'],
                      ['product_id','current_stock','reorder_point','p_sell']
                    ].sort_values('reorder_point', ascending=False)

    restock.to_csv('data/final/restock_list.csv', index=False)
    print(f"Identified {len(restock)} SKUs to reorder. Saved to data/final/restock_list.csv")

if __name__ == '__main__':
    main()
