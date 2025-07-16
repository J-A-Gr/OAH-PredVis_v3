#!/usr/bin/env python3
"""
train_assoc_plot_mapped.py

One-pager to load CSVs, split by order_date, run Apriori-style pair mining,
map product IDs to names, and plot the top N product pairs by lift.
"""

import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

# 1. CONFIG: adjust DATA_DIR if your data lives elsewhere
DATA_DIR       = "data/final"
ORDERS_CSV     = f"{DATA_DIR}/orders_with_coords_final_clean.csv"
ITEMS_CSV      = f"{DATA_DIR}/order_items_imputed.csv"
PRODUCTS_CSV   = f"{DATA_DIR}/products_imputed.csv"

# 2. PARAMETERS
MIN_SUPPORT   = 0.01   # minimum support threshold
TOP_N         = 10     # number of top pairs to plot
CUTOFF_DATE   = "2025-01-01"  # split train/test on this date

# 3. LOAD DATA
orders   = pd.read_csv(ORDERS_CSV,   parse_dates=["order_date"])
items    = pd.read_csv(ITEMS_CSV)
products = pd.read_csv(PRODUCTS_CSV)

# build mapping from product_id to human-readable name
# adjust 'product_name' if your products CSV uses a different column
id2name = products.set_index("product_id")["product_name"].to_dict()

# 4. PREPARE MERGED DF
df = (
    orders[["order_id", "order_date"]]
    .merge(items[["order_id", "product_id"]], on="order_id", how="inner")
)

# 5. SPLIT TRAIN / TEST
cutoff   = pd.Timestamp(CUTOFF_DATE)
train_df = df[df["order_date"] < cutoff]
print(f"Train orders: {train_df['order_id'].nunique()} unique orders")

# 6. COMPUTE PAIR COUNTS & METRICS
total_orders       = train_df["order_id"].nunique()
item_order_counts  = train_df.groupby("product_id")["order_id"].nunique().to_dict()
orders_to_products = train_df.groupby("order_id")["product_id"].apply(set)

pair_counts = {}
for products_set in orders_to_products:
    for a, b in combinations(sorted(products_set), 2):
        pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

records = []
for (a, b), count in pair_counts.items():
    support = count / total_orders
    if support < MIN_SUPPORT:
        continue
    confidence = count / item_order_counts[a]
    lift = support / (
        (item_order_counts[a] / total_orders) *
        (item_order_counts[b] / total_orders)
    )
    records.append({
        "antecedent": a,
        "consequent": b,
        "support": support,
        "confidence": confidence,
        "lift": lift
    })

pairs_df = pd.DataFrame(records)

# 7. MAP IDs TO NAMES & SELECT TOP N BY LIFT
pairs_df["antecedent_name"] = pairs_df["antecedent"].map(id2name)
pairs_df["consequent_name"] = pairs_df["consequent"].map(id2name)


# top_rules = pairs_df.sort_values("lift", ascending=False).head(TOP_N)

# print(f"Top {TOP_N} pairs by lift (names):")
# print(
#     top_rules[["antecedent_name","consequent_name","support","confidence","lift"]]
#     .to_string(index=False)
# )

# # 8. PLOT BAR CHART OF LIFT WITH NAMES
# labels = [
#     f"{row.antecedent_name} → {row.consequent_name}"
#     for _, row in top_rules.iterrows()
# ]

# plt.figure(figsize=(10,6))
# plt.bar(range(len(labels)), top_rules["lift"])
# plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
# plt.ylabel("Lift")
# plt.title(f"Top {TOP_N} Product Pairs by Lift (support ≥ {MIN_SUPPORT*100:.1f}%)")
# plt.tight_layout()
# plt.show()



sorted_rules = pairs_df.sort_values("lift", ascending=False)
start, end = TOP_N, TOP_N*2   # skip first 10, take next 10
next_rules = sorted_rules.iloc[start:end]

# print the “next 10”
print(f"Rules 11–20 by lift:")
print(
    next_rules[["antecedent_name","consequent_name","support","confidence","lift"]]
    .to_string(index=False)
)

# plot them
labels = [
    f"{row.antecedent_name} → {row.consequent_name}"
    for _, row in next_rules.iterrows()
]

plt.figure(figsize=(10,6))
plt.bar(range(len(labels)), next_rules["lift"])
plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
plt.ylabel("Lift")
plt.title(f"Rules 11–20 by Lift (support ≥ {MIN_SUPPORT*100:.1f}%)")
plt.tight_layout()
plt.show()



# sorted_rules = pairs_df.sort_values("lift", ascending=False)
# # skip first 20, take the next 10 (i.e. positions 20 through 29)
# start, end = TOP_N * 2, TOP_N * 3
# rules_21_30 = sorted_rules.iloc[start:end]

# print("Rules 21–30 by lift:")
# print(
#     rules_21_30[["antecedent_name","consequent_name","support","confidence","lift"]]
#     .to_string(index=False)
# )

# Plot them
# labels = [
#     f"{row.antecedent_name} → {row.consequent_name}"
#     for _, row in rules_21_30.iterrows()
# ]

# plt.figure(figsize=(10,6))
# plt.bar(range(len(labels)), rules_21_30["lift"])
# plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
# plt.ylabel("Lift")
# plt.title(f"Rules 21–30 by Lift (support ≥ {MIN_SUPPORT*100:.1f}%)")
# plt.tight_layout()
# plt.show()
