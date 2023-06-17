import pandas as pd
from apriori import Apriori
import numpy as np 
import time

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
df.replace({'y': 1, 'n': 0, '?':np.nan}, inplace=True)
df.dropna(inplace=True)

# Get the current column names
column_names = df.columns.tolist()

# Create a dictionary to store the new column names
new_column_names = {}

# Replace hyphens with underscores in column names
for column_name in column_names:
    new_column_name = column_name.replace('-', '_')
    new_column_names[column_name] = new_column_name

# Rename the columns using the new column names
df.rename(columns=new_column_names, inplace=True)

ap = Apriori(df, attribute_names, "class_name", 0.3)
start = time.time()
ap.generate()
end = time.time()
print("Time taken: ", end - start)
itemsets = ap.itemset

# Combine the itemsets into a single list
#itemsets = [item for sublist in itemsets for item in sublist]

values = []
num_keys = 0
for key, value in itemsets.items():
  #print(key)
  #print(len(value))
  num_keys += len(value)
  for i in value:
     
    print(i.items, "-", str(i.support))

print("Num rules", num_keys)

