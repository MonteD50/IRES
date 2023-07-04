import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics

import pandas as pd
import numpy as np
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt

"""
TODO: 
Spar
cmar
mac
all class association rule mining algorithms. Run and compare
"""

df = pd.read_csv("binary_distance_{'republican'}.csv", index_col=0)
# df = pd.read_csv("similarity_matrix_{'republican'}.csv", index_col=0)  # Uncomment this line if needed
df = df.astype(bool)
values = df.to_numpy()

cost = []
K = range(2, 10)

from kmodes.kmodes import KModes
from sklearn.metrics import silhouette_score

# Assuming your binary data is stored in a pandas DataFrame called df

# Define a range of cluster numbers to evaluate
min_clusters = 2
max_clusters = 10
K = range(min_clusters, max_clusters+1)
# Initialize variables to store the best silhouette score and corresponding cluster number
best_score = -1
best_clusters = -1

costs = []
silhouette_score_ = []
# Iterate through different cluster numbers and calculate the silhouette score
for k in range(min_clusters, max_clusters+1):
    km = KModes(n_clusters=k, init='Huang', n_init=5, verbose=0)
    clusters = km.fit_predict(df)
    score = silhouette_score(df, clusters, metric='matching')
    cost = km.cost_
    costs.append(cost)
    silhouette_score_.append(score)
    
    # Check if the current score is better than the previous best score
    if score > best_score:
        best_score = score
        best_clusters = k

# Print the best number of clusters and corresponding silhouette score
print("Best number of clusters:", best_clusters)
print("Silhouette score:", best_score)


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

# Fit KModes with the optimal number of clustersz
kmode = KModes(n_clusters=best_clusters, init="random", n_init=5, verbose=1)
clusters = kmode.fit_predict(values)

# Get the cluster centers
cluster_centers = kmode.cluster_centroids_

labels = kmode.labels_.tolist()
"""
max_clusters = 10
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
plt.title('K-means Clustering Results for Democrat')
plt.show()
"""