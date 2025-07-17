import pandas as pd
from sqlalchemy import create_engine
import os

# Make sure the folder exists
os.makedirs('data/final', exist_ok=True)

# Read your CSV
df = pd.read_csv('data/final/products_imputed.csv')

# Write to a SQLite file
engine = create_engine('sqlite:///data/final/products.db')
df.to_sql('products_imputed', engine, if_exists='replace', index=False)

print("SQLite DB written to data/final/products.db")
