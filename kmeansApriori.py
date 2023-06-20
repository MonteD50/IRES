import pandas as pd
import numpy as np

car = pd.read_csv("class_association_rules.csv")

result = {}
# Unique values for 'right'
unique_right = car['right'].unique()

column_names = car['left'].tolist()
# Make sure 'left' and 'right' are type sets
car['left'] = car['left'].apply(lambda x: set(x.replace("{", "").replace("}", "").replace("'", "").split(", ")))
car['right'] = car['right'].apply(lambda x: set(x.replace("{", "").replace("}", "").replace("'", "").split(", ")))

num_rows = car.shape[0]
# Store blank matrix of size num_rows x num_rows for each class
for class_name in unique_right:
    result[class_name] = np.zeros((num_rows, num_rows))

print("Nuum rows", num_rows)

for i in range(num_rows + 1):
    for j in range(i + 1, num_rows):
        row_i = car.iloc[i]
        row_j = car.iloc[j]
        if row_i['right'] == row_j['right']:
            simularity = 0

            left_rule_variables = set()
            right_rule_variables = set()
            for left_rule in row_i['left']:
                left_rule_value = left_rule.split("_")[-1]
                left_rule_variable = left_rule.replace("_" + left_rule_value, "")
                left_rule_variables.add(left_rule_variable)

                for right_rule in row_j['left']:
                    right_rule_value = right_rule.split("_")[-1]
                    right_rule_variable = right_rule.replace("_" + right_rule_value, "")
                    right_rule_variables.add(right_rule_variable)
                    
                    if left_rule_variable == right_rule_variable and left_rule_value != right_rule_value:
                        simularity += 2

            for left_rule in left_rule_variables:
                if left_rule not in right_rule_variables:
                    simularity += 1
            for right_rule in right_rule_variables:
                if right_rule not in left_rule_variables:
                    simularity += 1

            store_key = str(row_i['right'])
            result[store_key][i][j] = simularity
            

for key, value in result.items():
    # Flip the values so that the matrix is symmetrical
    value = value + value.T - np.diag(value.diagonal())
    df = pd.DataFrame(value, columns=column_names, index=column_names)
    df.to_csv("simularity_matrix_" + key + ".csv")
    break
    

            