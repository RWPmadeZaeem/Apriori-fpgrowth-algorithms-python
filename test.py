
import pandas as pd

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
df = pd.read_excel(url)

# Drop rows with missing values in the "InvoiceNo" column
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)

# Remove credit transactions (invoices starting with 'C')
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

# Remove spaces in the description column
df['Description'] = df['Description'].str.strip()

# Split the data by transaction (InvoiceNo)
basket = (df.groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

# Convert the quantity to binary (1 for items bought, 0 otherwise)
basket_encoded = basket.applymap(lambda x: 1 if x > 0 else 0)

# Step 3: Data Encoding
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
basket_encoded_te = te.fit_transform(basket_encoded)
df_encoded = pd.DataFrame(basket_encoded_te, columns=te.columns_)

# Step 4: Apply Apriori Algorithm
from mlxtend.frequent_patterns import apriori

frequent_itemsets_apriori = apriori(df_encoded, min_support=0.03, use_colnames=True)

# Step 5: Apply FP-Growth Algorithm
from mlxtend.frequent_patterns import fpgrowth

frequent_itemsets_fpgrowth = fpgrowth(df_encoded, min_support=0.03, use_colnames=True)

# Step 6: Analyze the Results
print("Apriori frequent itemsets:")
print(frequent_itemsets_apriori)

print("\nFP-Growth frequent itemsets:")
print(frequent_itemsets_fpgrowth)

# Step 7: Interpretation and Insights (Skipped in code)

# Step 8: Comparison and Evaluation
import time

# Measure execution time for Apriori
start_time = time.time()
apriori(df_encoded, min_support=0.03, use_colnames=True)
apriori_execution_time = time.time() - start_time

# Measure execution time for FP-Growth
start_time = time.time()
fpgrowth(df_encoded, min_support=0.03, use_colnames=True)
fpgrowth_execution_time = time.time() - start_time

print("\nExecution time for Apriori:", apriori_execution_time)
print("Execution time for FP-Growth:", fpgrowth_execution_time)
