import pandas as pd
import numpy as np
import time
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

attribute_names = [x.replace('-', '_') for x in attribute_names]

df2 = pd.read_csv("house-votes-84.data", names=attribute_names)
print(df2.shape)

"""
df2 = [
    ['republican', 'n', 'y', 'n', 'y', 'y', 'y', 'n', 'n', 'n', 'y', '?', 'y', 'y', 'y', 'n', 'y'],
    ['republican', 'n', 'y', 'n', 'y', 'y', 'y', 'n', 'n', 'n', 'n', 'n', 'y', 'y', 'y', 'n', '?'],
    ['democrat', '?', 'y', 'y', '?', 'y', 'y', 'n', 'n', 'n', 'n', 'y', 'n', 'y', 'y', 'n', 'n'],
    ['democrat', 'n', 'y', 'y', 'n', '?', 'y', 'n', 'n', 'n', 'n', 'y', 'n', 'y', 'n', 'n', 'y'],
    ['democrat', 'y', 'y', 'y', 'n', 'y', 'y', 'n', 'n', 'n', 'n', 'y', '?', 'y', 'y', 'y', 'y'],
    ['democrat', 'n', 'y', 'y', 'n', 'y', 'y', 'n', 'n', 'n', 'n', 'n', 'n', 'y', 'y', 'y', 'y'],
    ['democrat', 'n', 'y', 'n', 'y', 'y', 'y', 'n', 'n', 'n', 'n', 'n', 'n', '?', 'y', 'y', 'y'],
    ['republican', 'n', 'y', 'n', 'y', 'y', 'n', 'n', 'n', 'n', '?', '?', 'y', 'y', 'n', 'n', 'y'],
    ['republican', 'n', 'y', 'n', 'y', 'y', 'y', 'n', 'n', 'n', 'n', 'n', 'y', 'y', 'y', 'n', 'y'],
    ['democrat', 'y', 'y', 'y', 'n', 'n', 'n', 'y', 'y', 'y', 'n', 'n', 'n', 'n', 'n', '?', '?'],
    ['republican', 'n', 'y', 'n', 'y', 'y', 'n', 'n', 'n', 'n', 'n', '?', '?', 'y', 'y', 'n', '?'],
    ['republican', 'n', 'y', 'n', 'y', 'y', 'y', 'n', 'n', 'n', 'y', 'n', 'y', 'y', '?', 'n', '?'],
    ['democrat', 'n', 'y', 'y', 'n', 'n', 'n', 'y', 'y', 'y', 'n', 'n', 'n', 'y', 'n', '?', '?'],
    ['democrat', 'y', 'y', 'y', 'n', 'n', 'y', 'y', 'y', '?', 'y', 'y', '?', 'n', 'n', 'y', '?'],
    ['republican', 'n', 'y', 'n', 'y', 'y', 'y', 'n', 'n', 'n', 'n', 'n', 'y', '?', '?', 'n', '?'],
    ['republican', 'n', 'y', 'n', 'y', 'y', 'y', 'n', 'n', 'n', 'y', 'n', 'y', 'y', '?', 'n', '?'],
    ['democrat', 'y', 'n', 'y', 'n', 'n', 'y', 'n', 'y', '?', 'y', 'y', 'y', '?', 'n', 'n', 'y'],
    ['democrat', 'y', '?', 'y', 'n', 'n', 'n', 'y', 'y', 'y', 'n', 'n', 'n', 'y', 'n', 'y', 'y']
]
df2 = pd.DataFrame(df2, columns=attribute_names)

df2 = df2[["class_name", "handicapped_infants", "water_project_cost_sharing", "religious_groups_in_schools", "crime"]]
"""
print(df2.shape)

# Replace '?' with NaN
df2.replace('?', np.nan, inplace=True)

# Convert 'y' and 'n' to 1 and 0
#df2.replace({'y': 1, 'n': 0}, inplace=True)
#df.replace('?', 'unknown', inplace=True)

# Drop rows with missing values
df2.dropna(inplace=True)
print(df2.shape)
# Reset the index
df2.reset_index(drop=True, inplace=True)

# Drop the class_name column
df_without_class = df2.drop('class_name', axis=1, inplace=False)

df_encoded = pd.get_dummies(df_without_class)
df_encoded.to_csv("og_encoded.csv")
print(df_encoded.columns)
1/0

#df2 = df2[["class_name", "handicapped_infants", "water_project_cost_sharing",
#  "adoption_of_the_budget_resolution", "physician_fee_freeze",
#  "el_salvador_aid"]]

# Convert all rows to 1 or 0 factors
#df = df.astype('category')
#df_encoded = pd.get_dummies(df)

# Apply the Apriori algorithm
#start = time.time()
#frequent_itemsets = apriori(df_encoded, min_support=0.5, use_colnames=True)
#end = time.time()
# Sort by support
#frequent_itemsets.sort_values(by='support', ascending=False, inplace=True)
#print("Time elapsed: ", end - start)
#print(frequent_itemsets.shape)
#print(frequent_itemsets)

#frequent_itemsets.to_csv("frequent_itemsets_apriori.csv", index=False)
#print(frequent_itemsets)
# Generate association rules
#rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

# Display the association rules
#rules.to_csv("association_rules.csv", index=False)

# ========================== FP Growth Itemset ==========================
# FP Growth

from mlxtend.frequent_patterns import fpgrowth

start = time.time()
frequent_itemsets = fpgrowth(df_encoded, min_support=0.3, use_colnames=True)
end = time.time()
print(frequent_itemsets.shape)
print("Time elapsed: ", end - start)
frequent_itemsets.sort_values(by='support', ascending=False, inplace=True)
#frequent_itemsets.to_csv("frequent_itemsets_fpgrowth.csv", index=False)

# Add class_name to df_encoded
df_encoded['class_name'] = df2['class_name']
#df_encoded.to_csv("house-votes-84-encoded.csv", index=False)

# Class Association Rules
def class_assocation_rule(df_og, freq_itemsets, class_name, min_confidence):
    """
    Get the class association rules.

    Parameters
    ----------
    df_og : pandas DataFrame
        The dataset.
    freq_itemsets : pandas DataFrame
        The frequent itemsets.
    class_name : str
        The variable name of the class.
    min_confidence : float
        The minimum confidence.

    Returns
    -------
    pandas DataFrame
        The class association rules.
    """
    result = []
    num_rows = df_og.shape[0]
    classes = df_og[class_name].unique()
    for c in classes:
        for index, row in freq_itemsets.iterrows():
            leftside = set(row['itemsets'])
            rightside = {c}
            combined = leftside | rightside
            support = row['support']
            support_denominator = support * num_rows
            query_expression = f"{class_name} == '{c}' & "
            for item in leftside:
              query_expression = query_expression + item + " == 1 & "
            query_expression = query_expression[:-2]
            confidence = len(df_og.query(query_expression)) / support_denominator
            if confidence >= min_confidence:
                result.append({"left":leftside, "right":rightside, "support":support, "confidence": confidence})

    return pd.DataFrame(result)

start = time.time()

car = class_assocation_rule(df_encoded, frequent_itemsets, "class_name", min_confidence=0.4)
end = time.time()
print("Time elapsed: ", end - start)
# Sort 
car.sort_values(by='confidence', ascending=False, inplace=True)
print(car.shape)
car.to_csv("class_association_rules.csv", index=False)