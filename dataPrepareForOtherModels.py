import pandas as pd


df2 = pd.read_csv("Adult.csv")

# Replace all - column names with underscore
df2.columns = df2.columns.str.replace('-', '_')

# Replace '?' with NaN
#df2.replace('?', np.nan, inplace=True)

df2.dropna(inplace=True)

# Reset the index
df2.reset_index(drop=True, inplace=True)

# Drop the class_name column
df2 = df2.drop('id', axis=1, inplace=False)
df_without_class = df2.drop('class_name', axis=1, inplace=False)
df_encoded = pd.get_dummies(df_without_class)

# Add class_name column back to df_encoded
df_encoded['class_name'] = df2['class_name']

# Save to csv
df_encoded.to_csv('OtherModels/Adult_encoded.csv', index=False)