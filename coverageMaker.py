import pandas as pd
import warnings
warnings.filterwarnings('ignore')

clustered_rules = pd.read_csv("kmodes_distance_{'republican'}.csv")
df = pd.read_csv("og_encoded.csv")

clustered_rules['cluster'] = clustered_rules['cluster'].astype(int) + 2

unique_clusters = clustered_rules['cluster'].unique().tolist()

result = []
for cluster in unique_clusters:

    clustered_rules_cluster = clustered_rules[clustered_rules['cluster'] == cluster]

    for index, row in clustered_rules_cluster.iterrows():
        # Get all columns that are true in row
        true_columns = row[row == True].index.tolist()

        filtered_query = " == 1 & ".join(true_columns) + " == 1"
        
        # Filter the og dataset
        filtered_df = df.query(filtered_query)

        coverage = len(filtered_df.index) 
        
        clustered_rules_cluster.loc[index, "coverage"] = coverage

    # Sort by coverage
    clustered_rules_cluster = clustered_rules_cluster.sort_values(by=['coverage'], ascending=False)


    print(clustered_rules_cluster, "\n")

    start_index = 1
    get_top_coverage = 3
    current_num_coverage = 1

    res = []

    first_row = clustered_rules_cluster.iloc[0]
    res.append(first_row[first_row == True].index.tolist())

    while current_num_coverage < get_top_coverage:
        current_row = clustered_rules_cluster.iloc[start_index]
        prior_row = clustered_rules_cluster.iloc[start_index-1]
        # Get all columns that are true in row
        true_columns_current = current_row[current_row == True].index.tolist()
        true_columns_prior = prior_row[prior_row == True].index.tolist()

        filtered_query_current = " == 1 & ".join(true_columns_current) + " == 1"
        filtered_query_prior = " == 1 & ".join(true_columns_prior) + " == 1"
        
        # Filter the og dataset
        filtered_df_current = df.query(filtered_query_current).index.tolist()
        filtered_df_prior = df.query(filtered_query_prior).index.tolist()

        # Sort the lists
        filtered_df_current.sort()
        filtered_df_prior.sort()

        # Check if the lists are the same
        if filtered_df_current == filtered_df_prior:
            continue
        else:
            # Store the rule
            res.append(true_columns_current)
            current_num_coverage += 1

        start_index += 1
    print("wlkfhwelifhwei", start_index)

    print(res, cluster)
    result += res


print(result)
