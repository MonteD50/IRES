import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics

import pandas as pd
import numpy as np
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt

df = pd.read_csv("binary_distance_{'democrat'}.csv", index_col=0)
# df = pd.read_csv("similarity_matrix_{'republican'}.csv", index_col=0)  # Uncomment this line if needed

values = df.to_numpy()

cost = []
K = range(2, 10)

for k in K:
    kmode = KModes(n_clusters=k, init="Huang", n_init=5, verbose=1)
    kmode.fit_predict(values)
    cost.append(kmode.cost_)

plt.plot(K, cost, 'x-')
plt.xlabel('Number of clusters')
plt.ylabel('Cost')
plt.title('Elbow Curve')
plt.show()

# Fit KModes with the optimal number of clusters
optimal_k = np.argmin(cost) + 1
kmode = KModes(n_clusters=optimal_k, init="random", n_init=5, verbose=1)
clusters = kmode.fit_predict(values)

# Get the cluster centers
cluster_centers = kmode.cluster_centroids_

print(kmode.labels_)


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