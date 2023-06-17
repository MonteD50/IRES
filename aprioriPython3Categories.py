"""
Expirement: Apriori Algorithm on the House Voting Dataset
Testing number of frequent itemsets for having 3 categories: y, n, unknown
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori


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


# Replace '?' with NaN
#df.replace('?', np.nan, inplace=True)


# Convert 'y' and 'n' to 1 and 0
#df.replace({'y': 1, 'n': 0, '?': 2}, inplace=True)
df.replace('?', 'unknown', inplace=True)

# Drop rows with missing values
#df.dropna(inplace=True)


# Reset the index
#df.reset_index(drop=True, inplace=True)


# Drop the class_name column
df.drop('class_name', axis=1, inplace=True)


# Convert all rows to 1 or 0 factors
#df = df.astype('category')
df_encoded = pd.get_dummies(df)
# Get unique values of each column
unique_values = df_encoded.nunique()
#df_encoded.to_csv("test.csv", index=False)
print(unique_values)
# Apply the Apriori algorithm
frequent_itemsets = apriori(df_encoded, min_support=0.3, use_colnames=True)
print(frequent_itemsets.shape)
#frequent_itemsets.to_csv("frequent_itemsets_3categories.csv", index=False)

# Generate association rules
#rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)


# Display the association rules
#rules.to_csv("association_rules.csv", index=False)
