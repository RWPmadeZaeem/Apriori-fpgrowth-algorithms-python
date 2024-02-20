# Import necessary libraries
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

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

# Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)

# Calculate the length of each itemset
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

# Filter the frequent itemsets based on length
filtered_itemsets = frequent_itemsets[frequent_itemsets['length'] >  1]

# Extract association rules with a high confidence level
rules = association_rules(filtered_itemsets, metric="confidence", min_threshold=0.6, support_only= True)

# Display the rules
print(rules)
