# Import necessary libraries
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Example transactions
transactions = [['milk', 'bread', 'butter'],
                ['milk', 'bread', 'butter'],
                ['milk', 'bread'],
                ['milk', 'butter'],
                ['milk', 'butter'],
                ['milk', 'butter'],
                ['milk', 'bread', 'butter', 'beer']]

# Encode the dataset into a one-hot encoded DataFrame
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Apply the FP-Growth algorithm to find frequent itemsets
frequent_itemsets = fpgrowth(df, min_support=0.6, use_colnames=True)

# Optionally, calculate the length of each itemset
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

# Optionally, filter the frequent itemsets based on certain criteria
filtered_itemsets = frequent_itemsets[frequent_itemsets['length'] > 1]

# Extract association rules with a high confidence level
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Display the rules
print(rules)
