import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from prophet import Prophet

# 1) Association Rules
basket = pd.read_csv('data/work_material/basket_df.csv', index_col=0)
# find itemsets with ≥1% support
freqs = apriori(basket, min_support=0.01, use_colnames=True)
rules = association_rules(freqs, metric="lift", min_threshold=1.0)
top_rules = (
    rules.sort_values('lift', ascending=False)
         .head(10)
         [['antecedents','consequents','support','confidence','lift']]
)
top_rules.to_csv('data/work_material/top_rules.csv', index=False)
print("✅ Top 10 rules written to data/work_material/top_rules.csv")

# 2) Time-Series Forecast
ts = pd.read_csv('data/work_material/ts_df.csv', parse_dates=['order_date'])
daily = ts.groupby('order_date')['quantity'].sum().reset_index()
daily = daily.rename(columns={'order_date':'ds','quantity':'y'})

m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
m.fit(daily)

future = m.make_future_dataframe(periods=7)
forecast = m.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(7) \
        .to_csv('data/work_material/7day_forecast.csv', index=False)
print("✅ 7-day forecast written to data/work_material/7day_forecast.csv")
