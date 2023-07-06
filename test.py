import pandas as pd


df = pd.read_csv("Nursary.csv")

# Count number of values in class_name
print(df['class_name'].value_counts())