import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np
from datetime import timedelta

# Load your time-series data
ts_df = pd.read_csv('data/work_material/ts_df.csv', parse_dates=['order_date'])

# Set up hold-out: last 30 days
last_date = ts_df['order_date'].max()
holdout_start = last_date - timedelta(days=30)

# Pick top 5 SKUs by total units sold
top_skus = (
    ts_df.groupby('product_id')['net_units']
         .sum()
         .nlargest(5)
         .index
         .tolist()
)

results = []
for sku in top_skus:
    df_sku = (
        ts_df[ts_df['product_id'] == sku]
        .set_index('order_date')
        .asfreq('D', fill_value=0)
        .reset_index()
    )
    train = df_sku[df_sku['order_date'] <= holdout_start]
    test  = df_sku[df_sku['order_date'] >  holdout_start]

    # Prepare for Prophet
    train_prophet = train.rename(columns={'order_date':'ds','net_units':'y'})[['ds','y']]
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.fit(train_prophet)

    # Forecast for hold‐out period
    future = m.make_future_dataframe(periods=len(test), freq='D')
    fcst = m.predict(future).set_index('ds')['yhat']
    
    # Evaluate
    y_true = test['net_units'].values
    y_pred = fcst.loc[test['order_date']].values
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    results.append({'product_id': sku, 'MAPE': mape, 'RMSE': rmse})

# Tabulate
print(pd.DataFrame(results))
