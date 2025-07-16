import pandas as pd

# 1) Load your full orders dataset
df = pd.read_csv('data/imputed/orders_imputed.csv', parse_dates=['CreatedOn'])

# 2) Extract unique (city, country) pairs, drop any incomplete rows
unique_regions = (
    df[['DeliveryCity', 'DeliveryCountry']]
    .dropna(subset=['DeliveryCity', 'DeliveryCountry'])
    .drop_duplicates()
    .assign(
        DeliveryCity=lambda d: d['DeliveryCity'].str.strip(),    # normalize whitespace
        DeliveryCountry=lambda d: d['DeliveryCountry'].str.strip()
    )
)

# 3) Add placeholder columns for latitude & longitude
unique_regions['lat'] = ''
unique_regions['lon'] = ''

# 4) Save to CSV
output_path = 'data/region_coords.csv'
unique_regions.to_csv(output_path, index=False)

print(f"Extracted {len(unique_regions)} unique city–country pairs → {output_path}")
