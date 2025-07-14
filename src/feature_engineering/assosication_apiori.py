import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Load your precomputed baskets
basket_df = pd.read_csv('data/work_material/basket_df.csv')

# Convert the list-strings back into actual Python lists
basket_df['products'] = basket_df['products'].apply(eval)

# Transaction encoder
te = TransactionEncoder()
te_ary = te.fit(basket_df['products']).transform(basket_df['products'])
trans_df = pd.DataFrame(te_ary, columns=te.columns_)

# Mine frequent item sets and rules
freq_itemsets = apriori(trans_df, min_support=0.01, use_colnames=True)
rules = association_rules(freq_itemsets, metric="lift", min_threshold=1.2)

# Show top 10 by lift
top_rules = rules.sort_values('lift', ascending=False).head(10)
print(top_rules[['antecedents','consequents','support','confidence','lift']])
