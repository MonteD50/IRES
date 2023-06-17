import pandas as pd
import numpy as np
import time
import Orange
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


attribute_names = [
  "class_name", "handicapped-infants", "water-project-cost-sharing",
  "adoption-of-the-budget-resolution", "physician-fee-freeze",
  "el-salvador-aid", "religious-groups-in-schools", "anti-satellite-test-ban",
  "aid-to-nicaraguan-contras", "mx-missile", "immigration",
  "synfuels-corporation-cutback", "education-spending",
  "superfund-right-to-sue", "crime", "duty-free-exports",
  "export-administration-act-south-africa"
]

df = pd.read_csv("house-votes-84.data", names=attribute_names)
print(df.shape)

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Convert 'y' and 'n' to 1 and 0
df.replace({'y': 1, 'n': 0}, inplace=True)
#df.replace('?', 'unknown', inplace=True)

# Drop rows with missing values
df.dropna(inplace=True)
print(df.shape)
# Reset the index
df.reset_index(drop=True, inplace=True)

# Drop the class_name column
df.drop('class_name', axis=1, inplace=True)

# Convert all rows to 1 or 0 factors
df = df.astype('category')
#df_encoded = pd.get_dummies(df)

# Apply the Apriori algorithm
start = time.time()
frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
end = time.time()
# Sort by support
frequent_itemsets.sort_values(by='support', ascending=False, inplace=True)
print("Time elapsed: ", end - start)
print(frequent_itemsets.shape)
print(frequent_itemsets)
# Filter by support == 0.11637931034482758
#frequent_itemsets = frequent_itemsets[frequent_itemsets['support'] == 0.11637931034482758]
frequent_itemsets.to_csv("frequent_itemsets_apriori.csv", index=False)
#print(frequent_itemsets)
# Generate association rules
#rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

# Display the association rules
#rules.to_csv("association_rules.csv", index=False)

# ========================== FP Growth Itemset ==========================
# FP Growth

from mlxtend.frequent_patterns import fpgrowth

start = time.time()
frequent_itemsets = fpgrowth(df, min_support=0.1, use_colnames=True)
end = time.time()
print("Time elapsed: ", end - start)
frequent_itemsets.sort_values(by='support', ascending=False, inplace=True)
frequent_itemsets.to_csv("frequent_itemsets_fpgrowth.csv", index=False)

