import pandas as pd
import numpy as np
from kmodes.kmodes import KModes
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import hamming
from mlxtend.frequent_patterns import fpgrowth
import warnings
warnings.filterwarnings('ignore')


class ClassAssRules:
    def __init__(self, df_og,
                 df_encoded,
                 # test_set, 
                 target_class: list,
                 target_class_name: str,
                 get_top_coverage = 3,
                 min_support = 0.3,
                 min_confidence = 0.4, 
                 max_rules = 1000,
                 max_clusters = 10,
                 num_clusters = None,
                 graph = True,
                 method = 'kmodes',
                 cars = None):
        
        self.df_og = df_og
        self.df_encoded = df_encoded

        self.target_class = target_class
        self.target_class_name = target_class_name
        self.get_top_coverage = get_top_coverage
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.max_rules = max_rules
        self.max_clusters = max_clusters
        self.graph = graph
        self.method = method
        self.cars = cars # csv file of prior made class association rules

        # Num clusters should be a dictionary of {class_name: num clusters} or None
        self.num_clusters = num_clusters

        self.variables = list(self.df_encoded.columns)

    
    def predict(self, test_set_without_target):
        for index, row in test_set_without_target.iterrows():
            prediction_count = {}
            for class_name, rules in self.rules.items():
                prediction_count[class_name] = 0
                for rule in rules:
                    # Get all columns that are true in row
                    true_columns = row[row == True].index.tolist()

                    # If the rule is a subset of true_columns
                    if set(rule) <= set(true_columns):
                        prediction_count[class_name] += 1
                        
                prediction_count[class_name] /= len(rules)
                #test_set_without_target.loc[index, f"prediction_count_for_{class_name}"] = prediction_count

            # Get the max prediction_count
            prediction = max(prediction_count, key=prediction_count.get)
            prediction_count_value = max(prediction_count.values())

            for class_name, count in prediction_count.items():
                test_set_without_target.loc[index, f"prediction_count_for_{class_name}"] = count

            # Store
            test_set_without_target.loc[index, "prediction"] = prediction
            test_set_without_target.loc[index, "prediction_count"] = prediction_count_value

        return test_set_without_target


    def run(self):
        self.rules = {}
        # ================== Generate class association rules ===============
        print("Starting cars")
        if self.cars is not None:
            self.car = pd.read_csv(self.cars)
        else:
            self._generate_cars()

        # ================== Generate distances ======================
        print("Generating distnaces")
        if self.method == 'kmodes':
            self._generate_binary_distance()
        else:
            self._generate_simularity_rule_distance()

        # =================== Clustering =====================
        for class_name, distances in self.distances.items():
            class_name = class_name.replace("{", "").replace("}", "").replace("'", "")
            print("Clustering:", class_name)
            if self.num_clusters is not None:
                if class_name in self.num_clusters:

                    clustered_df = self._cluster(distances, 
                                                        self.num_clusters[class_name])
                else:
                    print("WARNING: The class you specified is not a valid class name")
                    print("These are the valid class names:", self.distances.keys())
                    clustered_df = self._cluster(distances)
            else:
                clustered_df = self._cluster(distances)

            # ================== Coverage ======================
            print("Coverage")
            rules_per_class = self._get_rules_by_coverage(clustered_df)

            self.rules[class_name] = rules_per_class
        #print(self.rules)


    def _cluster(self, distances, num_clusters = None):
        if self.method == 'kmodes':
            print("Running KModes")
            return self._cluster_kmodes(distances, num_clusters)
        else:
            print("Running Kmeans")
            return self._cluster_kmeans(distances, num_clusters)


    def _cluster_kmeans(self, distances, num_clusters = None):
        column_names = self.car['left'].astype(str).tolist()

        value = distances + distances.T - np.diag(distances.diagonal())
        value = value.astype(int)

        df = pd.DataFrame(value, columns=column_names, index=column_names) 

        if num_clusters is None:
            min_clusters = 2
            K = range(min_clusters, self.max_clusters+1)

            # Initialize variables to store the best ch index score and corresponding cluster number
            best_score = -1
            best_clusters = -1
            ch_index_ = []
            # Iterate through different cluster numbers and calculate the ch index
            for k in range(min_clusters, self.max_clusters+1):
                kmeans = KMeans(n_clusters=k, random_state=0).fit(value)
                labels = kmeans.labels_
                ch = metrics.calinski_harabasz_score(value, labels)
                if ch > best_score:
                    best_score = ch
                    best_clusters = k
                ch_index_.append(ch)

            print("Best number of clusters:", best_clusters)
            print("CH index", best_score)

            if self.graph:
                plt.plot(K, ch_index_, 'x-')
                plt.xlabel('Number of clusters')
                plt.ylabel('CH INdex Score')
                plt.title('CH Index Score Curve')
                plt.show()

        else:
            best_clusters = num_clusters

        # Get the cluster labels assigned to each data point
        kmeans = KMeans(n_clusters=best_clusters, random_state=0).fit(value)
        labels = kmeans.labels_
        
        # Get the cluster centers
        centers = kmeans.cluster_centers_

        df['cluster'] = labels

        # Foreach row calculate the distance to the cluster center
        for index, row in df.iterrows():
            cluster = row['cluster']
            cluster_center = centers[cluster]
            df.loc[index, "distance"] = np.linalg.norm(row[:-1] - cluster_center)
 
        return df

    def _get_rules_by_coverage(self, clustered_rules):
        clustered_rules['cluster'] = clustered_rules['cluster'].astype(int) + 2

        unique_clusters = clustered_rules['cluster'].unique().tolist()

        unique_clusters.sort()
        result = []
        for cluster in unique_clusters:
            clustered_rules_cluster = clustered_rules[clustered_rules['cluster'] == cluster]

            for index, row in clustered_rules_cluster.iterrows():
                # Get all columns that are true in row
                true_columns = row[row == True].index.tolist()

                filtered_query = " == 1 & ".join(true_columns) + " == 1"

                # Filter the og dataset
                filtered_df = self.df_encoded.query(filtered_query)

                coverage = len(filtered_df.index)

                clustered_rules_cluster.loc[index, "coverage"] = coverage

            # Sort by coverage and distance
            clustered_rules_cluster = clustered_rules_cluster.sort_values(by=['coverage', 'distance'], ascending=[False, True])

            start_index = 1
            current_num_coverage = 1

            res = []

            first_row = clustered_rules_cluster.iloc[0]
            res.append(first_row[first_row == True].index.tolist())

            while current_num_coverage < self.get_top_coverage and start_index < len(clustered_rules_cluster.index):
                current_row = clustered_rules_cluster.iloc[start_index]
                prior_row = clustered_rules_cluster.iloc[start_index-1]

                start_index += 1
                # Get all columns that are true in row
                true_columns_current = current_row[current_row == True].index.tolist()
                true_columns_prior = prior_row[prior_row == True].index.tolist()

                filtered_query_current = " == 1 & ".join(true_columns_current) + " == 1"
                filtered_query_prior = " == 1 & ".join(true_columns_prior) + " == 1"

                # Filter the og dataset
                filtered_df_current = self.df_encoded.query(filtered_query_current).index.tolist()
                filtered_df_prior = self.df_encoded.query(filtered_query_prior).index.tolist()

                # Sort the lists
                filtered_df_current.sort()
                filtered_df_prior.sort()

                # Check if the lists are the same
                if filtered_df_current == filtered_df_prior:

                    continue
                else:
                    # Check if the rule is unique based on coverage to every
                    # prior rule that has been made
                    is_unique = True
                    for rule_ in result:
                        filtered_query_prior_rule = " == 1 & ".join(rule_) + " == 1"
                        filtered_df__prior_rule = self.df_encoded.query(filtered_query_prior_rule).index.tolist()

                        filtered_df__prior_rule.sort()
                        if filtered_df_current == filtered_df__prior_rule:
                            is_unique = False
                            break

                    if is_unique:
                        # Store the rule
                        res.append(true_columns_current)
                        current_num_coverage += 1

                
            result += res

        return result



    def _cluster_kmodes(self, distance_matrix, num_clusters = None):
        df = distance_matrix.astype(bool)
        values = df.to_numpy()

        if num_clusters is None:
            cost = []
            min_clusters = 2
            K = range(min_clusters, self.max_clusters+1)
            # Initialize variables to store the best silhouette score and corresponding cluster number
            best_score = -1
            best_clusters = 2
            costs = []
            silhouette_score_ = []
            # Iterate through different cluster numbers and calculate the silhouette score
            for k in range(min_clusters, self.max_clusters+1):
                try:
                    km = KModes(n_clusters=k, init='Huang', n_init=5, verbose = 0)
                    clusters = km.fit_predict(df)
                    score = silhouette_score(df, clusters, metric='matching')
                    cost = km.cost_
                    costs.append(cost)
                    silhouette_score_.append(score)
                    
                    # Check if the current score is better than the previous best score
                    if score > best_score:
                        best_score = score
                        best_clusters = k
                except Exception as e:
                    print("Error in optimal clusters kmodes", str(e))

            # Print the best number of clusters and corresponding silhouette score
            print("Best number of clusters:", best_clusters)
            print("Silhouette score:", best_score)

            if self.graph:
                plt.plot(K, costs, 'x-')
                plt.xlabel('Number of clusters')
                plt.ylabel('Cost')
                plt.title('Elbow Curve')
                plt.show()

                plt.plot(K, silhouette_score_, 'x-')
                plt.xlabel('Number of clusters')
                plt.ylabel('Silhouette Score')
                plt.title('Silhouette Score Curve')
                plt.show()

        else:
            best_clusters = num_clusters

        # Fit KModes with the optimal number of clusters
        try:
            kmode = KModes(n_clusters=best_clusters, init="random", n_init=5, verbose=0)
            clusters = kmode.fit_predict(values)
    
            cluster_centers = kmode.cluster_centroids_

            labels = kmode.labels_.tolist()

            # Add labels to the dataframe
            df['cluster'] = labels

            # For each row in df, calculate hammings distance from the cluster center
            df['distance'] = df.apply(lambda row: self._calc_hamming(row, cluster_centers[row['cluster']]), axis=1)

        except Exception as e:
            print("Error here", str(e))
            df['cluster'] = 0
            df['distance'] = 0

        return df
    
    def _calc_hamming(self, one, two):
        one = one.drop(labels=['cluster']).values
        return hamming(one, two)
        

    def _generate_simularity_rule_distance(self):
        self.distances = {}
        # Unique values for 'right'
        self.car['right'] = self.car['right'].astype(str)
        unique_right = self.car['right'].unique()

        #column_names = self.car['left'].tolist()
        # Make sure 'left' and 'right' are type sets
        #self.car['left'] = self.car['left'].apply(lambda x: set(x.replace("{", "").replace("}", "").replace("'", "").split(", ")))
        #self.car['right'] = self.car['right'].apply(lambda x: set(x.replace("{", "").replace("}", "").replace("'", "").split(", ")))

        num_rows = self.car.shape[0]
        # Store blank matrix of size num_rows x num_rows for each class
        for class_name in unique_right:
            self.distances[class_name] = np.zeros((num_rows, num_rows))

        for i in range(num_rows + 1):
            for j in range(i + 1, num_rows):
                row_i = self.car.iloc[i]
                row_j = self.car.iloc[j]

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
                    self.distances[store_key][i][j] = simularity 


    def _generate_binary_distance(self):
        # Unique values for 'right'
        self.car['right'] = self.car['right'].astype(str)
        unique_right = self.car['right'].unique()

        self.distances = {}
        for class_name in unique_right:
            # Convert to set
            #class_name = eval(class_name)
            df_class = self.car[self.car['right'] == class_name]
            res = pd.DataFrame(columns = self.variables)
            # Loop through df_class
            for index, row in df_class.iterrows():
                left = row['left']
                #left = eval(left)
                res_row = {}
                for variable in self.variables:
                    if variable in left:
                        label = 1
                    else:
                        label = 0
                    res_row[variable] = label
                left = str(left)
                row_series = pd.Series(res_row, name = left)
                res = res.append(row_series, ignore_index = False)
            
            self.distances[class_name] = res
            

    def _generate_cars(self):
        # Generate frequent itemsets
        frequent_itemsets = fpgrowth(self.df_encoded, 
                                     min_support=self.min_support, 
                                     use_colnames=True)
        
        print("ok")
 
        frequent_itemsets.sort_values(by='support', ascending=False, inplace=True)

        frequent_itemsets = frequent_itemsets.head(self.max_rules)
        
        # Add class_name to df_encoded
        self.df_encoded['class_name'] = self.target_class

        # Generate class association rules
        self.car = self.class_assocation_rule(self.df_encoded, 
                                         frequent_itemsets, 
                                         self.target_class_name, 
                                         min_confidence=self.min_confidence)
        
        # Sort 
        self.car.sort_values(by=['confidence', 'support'], ascending=[False, False], inplace=True)


    def class_assocation_rule(self, df_og, freq_itemsets, class_name, min_confidence):
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
            #c = int(c) # Temp only for dna
            print(c, len(freq_itemsets))
            for index, row in freq_itemsets.iterrows():
                leftside = set(row['itemsets'])
                rightside = {c}
                combined = leftside | rightside
                support = row['support']
                support_denominator = support * num_rows
                query_expression = f"{class_name} == '{c}' & "
                #print(leftside)
                for item in leftside:
                    query_expression = query_expression + item + " == 1 & "
                query_expression = query_expression[:-2]
                #print(query_expression)
                confidence = len(df_og.query(query_expression)) / support_denominator
                
                if confidence >= min_confidence:
                    result.append({"left":leftside, "right":rightside, "support":support, "confidence": confidence})

        return pd.DataFrame(result)