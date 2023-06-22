import pandas as pd
import numpy as np
import time

# TODO: seperate out the classes that are not equivalent (don't include them in the matrix)
car = pd.read_csv("class_association_rules.csv")

# Group by right and get 10 from each
#car = car.groupby('right').head(10)
# Get {'democrat'}
car = car[car['right'] == "{'democrat'}"].head(10)

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

start = time.time()
for i in range(num_rows + 1):
    if i % 100 == 0:
        print(i, "Time elapsed: ", time.time() - start)
        start = time.time()
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
            
print("Done=== starting kmeans")
for key, value in result.items():
    value = value + value.T - np.diag(value.diagonal())
    value = value.astype(int)
    df = pd.DataFrame(value, columns=column_names, index=column_names)
    df.to_csv("simularity_matrix_" + key + ".csv")


"""
max_clusters = 20
for key, value in result.items():
    # Flip the values so that the matrix is symmetrical
    value = value + value.T - np.diag(value.diagonal())
    value = value.astype(int)
    
    max_ch = 0
    best_k = 1
    chs = []
    for i in range(2, max_clusters):
        kmeans = KMeans(n_clusters=i, random_state=0).fit(value)

        labels = kmeans.labels_

        ch = metrics.calinski_harabasz_score(value, labels)
        if ch > max_ch:
            max_ch = ch
            best_k = i
        chs.append(ch)
    print(chs)
    #df = pd.DataFrame(value, columns=column_names, index=column_names)
    # Get the cluster labels assigned to each data point
    kmeans = KMeans(n_clusters=best_k, random_state=0).fit(value)
    print(best_k, max_ch)
    labels = kmeans.labels_
   

    # Get the cluster centers
    centers = kmeans.cluster_centers_

    # Visualize the data points and cluster centers
    plt.scatter(value[:, 0], value[:, 1], c=labels)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'K-means Clustering Results for {key}')
    plt.show()
    break
    #df.to_csv("simularity_matrix_" + key + ".csv")
"""

            