import pandas as pd

df = pd.read_csv('data/master_transactions.csv')

print(df["DeliveryCountry"].notnull())