import pandas as pd

# Adjust the file path as needed
file_path = '/Users/taief/Desktop/MusicTry/data_by_artist.csv'
data_by_artist = pd.read_csv(file_path)

data_by_artist.head()

# ----------------------------------------
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Selecting the features to use for clustering
features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'popularity']
data_selected = data_by_artist[features]

# Dropping rows with missing values
data_clean = data_selected.dropna()

# Data normalization
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_clean)

# Convert the scaled data back to a DataFrame (optional, for convenience)
data_scaled_df = pd.DataFrame(data_scaled, columns=features)

# ----------------------------------------
from sklearn.decomposition import PCA

# Step 1: PCA for Dimensionality Reduction to retain 90% of the variance
pca = PCA(n_components=0.90)
data_pca = pca.fit_transform(data_scaled_df)

# Now, data_pca contains the reduced dataset with the principal components that account for at least 90% of the total variance.

# You can check how many components were selected to meet this variance threshold:
print(f"Number of components selected to retain 90% variance: {pca.n_components_}")

#print variance
print(pca.explained_variance_ratio_)
# ----------------------------------------
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Step 2: Perform Hierarchical Clustering on the PCA-reduced data
Z = linkage(data_pca, method='ward')

# Step 3: Plot the Hierarchical Clustering Dendrogram
plt.figure(figsize=(10, 7))
plt.title('Hierarchical Clustering Dendrogram (PCA-reduced Data)')
dendrogram(Z, truncate_mode='lastp', p=12, leaf_rotation=45., leaf_font_size=15., show_contracted=True)
plt.xlabel('Cluster Size')
plt.ylabel('Distance')
plt.show()

# ----------------------------------------
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Perform Hierarchical Clustering with 'complete' linkage
Z_complete = linkage(data_pca, method='complete')
plt.figure(figsize=(10, 7))
plt.title('Hierarchical Clustering Dendrogram (Complete Linkage)')
dendrogram(Z_complete, truncate_mode='lastp', p=12, leaf_rotation=45., leaf_font_size=15., show_contracted=True)
plt.xlabel('Cluster Size')
plt.ylabel('Distance')
plt.show()


# You have already generated the dendrogram with 'ward' linkage, so it's not included here.
# ----------------------------------------
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Perform Hierarchical Clustering with 'complete' linkage
Z_average = linkage(data_pca, method='average')
plt.figure(figsize=(10, 7))
plt.title('Hierarchical Clustering Dendrogram (Average Linkage)')
dendrogram(Z_average, truncate_mode='lastp', p=12, leaf_rotation=45., leaf_font_size=15., show_contracted=True)
plt.xlabel('Cluster Size')
plt.ylabel('Distance')
plt.show()


# ----------------------------------------
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fcluster

def calculate_silhouette_scores(data, linkage_matrix, max_clusters=10):
    scores = []
    for num_clusters in range(2, max_clusters+1):
        # Assign cluster labels
        labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
        # Calculate silhouette score
        score = silhouette_score(data, labels, metric='euclidean')
        scores.append((num_clusters, score))
    return scores

# Calculate silhouette scores for each linkage method
ward_scores = calculate_silhouette_scores(data_pca, Z)
complete_scores = calculate_silhouette_scores(data_pca, Z_complete)
average_scores = calculate_silhouette_scores(data_pca, Z_average)

# Print silhouette scores for inspection
print("Ward linkage scores:", ward_scores)
print("Complete linkage scores:", complete_scores)
print("Average linkage scores:", average_scores)

# ----------------------------------------
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Range of possible clusters to evaluate
range_n_clusters = list(range(2, 11))

silhouette_scores_Kmeans = []  # To store silhouette scores for each n_clusters

for n_clusters in range_n_clusters:
    # Initialize KMeans with n_clusters
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(data_pca)  # Use PCA-reduced data
    
    # Calculate the silhouette score and append to list
    silhouette_avg = silhouette_score(data_pca, cluster_labels)
    silhouette_scores_Kmeans.append(silhouette_avg)
    print(f"For n_clusters = {n_clusters}, the silhouette score is: {silhouette_avg}")

# Plotting the silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, silhouette_scores_Kmeans, marker='o')
plt.title('Silhouette Score for Different Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()


optimal_clusters = range_n_clusters[silhouette_scores_Kmeans.index(max(silhouette_scores_Kmeans))]
print(f"The optimal number of clusters is: {optimal_clusters}")
# ----------------------------------------
import numpy as np
import hdbscan
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Define a range for the minimum cluster size you want to test
min_cluster_sizes = range(2, 12)  # Example: from 2 to 11

silhouette_scores_hdbscan = []

for size in min_cluster_sizes:
    # Initialize HDBSCAN with varying min_cluster_size
    clusterer_hdbscan = hdbscan.HDBSCAN(min_cluster_size=size, min_samples=1, gen_min_span_tree=True)
    cluster_labels_hdbscan = clusterer_hdbscan.fit_predict(data_pca)
    
    # Only calculate silhouette score if more than one cluster is found and not all points are noise
    if len(np.unique(cluster_labels_hdbscan)) > 1 and not np.all(cluster_labels_hdbscan == -1):
        silhouette_avg_hdbscan = silhouette_score(data_pca, cluster_labels_hdbscan)
        silhouette_scores_hdbscan.append(silhouette_avg_hdbscan)
        print(f"Min cluster size = {size}, Silhouette Score = {silhouette_avg_hdbscan}")
    else:
        # Handle the case where HDBSCAN finds only one cluster or all points are noise
        silhouette_scores_hdbscan.append(-1)  # Indicative value to show poor clustering
        print(f"Min cluster size = {size}, unable to calculate silhouette score (insufficient clusters).")

# Plotting the silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(min_cluster_sizes, silhouette_scores_hdbscan, marker='o')
plt.title('Silhouette Scores for Different Min Cluster Sizes (HDBSCAN)')
plt.xlabel('Min Cluster Size')
plt.ylabel('Silhouette Score')
plt.show()

# ----------------------------------------
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

# Prepare your data: ensure data_pca is your PCA-reduced dataset ready for clustering
# data_pca = ...

# Define ranges for the parameters you want to test
xi_values = np.linspace(0.01, 0.1, 5)  # Adjust the range and number of steps as needed
min_cluster_sizes = range(2, 12)  # For example, from 2 to 11

# Initialize lists to store the results
results = []

# Loop over all combinations of xi and min_cluster_size
for xi in xi_values:
    for min_cluster_size in min_cluster_sizes:
        # Fit the OPTICS model with the current set of parameters
        optics_model = OPTICS(xi=xi, min_cluster_size=min_cluster_size)
        optics_model.fit(data_pca)
        
        # Extract labels
        labels = optics_model.labels_
        
        # Calculate the silhouette score, ensuring we have more than one cluster and not all are noise
        if len(set(labels)) > 1 and not np.all(labels == -1):
            silhouette_avg = silhouette_score(data_pca, labels)
            results.append((xi, min_cluster_size, silhouette_avg))
            print(f"xi: {xi:.2f}, min_cluster_size: {min_cluster_size}, Silhouette Score: {silhouette_avg:.4f}")
        else:
            print(f"xi: {xi:.2f}, min_cluster_size: {min_cluster_size}, No valid clusters formed.")

# Analyze the results
# Convert results to a structured array for easier analysis
dtype = [('xi', float), ('min_cluster_size', int), ('silhouette_score', float)]
results_array = np.array(results, dtype=dtype)

# Sort results by silhouette score
sorted_results = np.sort(results_array, order='silhouette_score')

# Print the top configurations
print("Top configurations:")
for result in sorted_results[::-1][:5]:  # Adjust the number of top results to display as needed
    print(f"xi: {result['xi']}, min_cluster_size: {result['min_cluster_size']}, Silhouette Score: {result['silhouette_score']:.4f}")

# Optional: Plot the results
# This section assumes you want to visualize how silhouette scores vary with the parameters
# You might need to adapt this depending on how you want to visualize the results.

# ----------------------------------------
# plot these scores
import matplotlib.pyplot as plt

ward_scores_df = pd.DataFrame(ward_scores, columns=['Number of Clusters', 'Silhouette Score'])
complete_scores_df = pd.DataFrame(complete_scores, columns=['Number of Clusters', 'Silhouette Score'])
average_scores_df = pd.DataFrame(average_scores, columns=['Number of Clusters', 'Silhouette Score'])
optimal_clusters = range_n_clusters[silhouette_scores_Kmeans.index(max(silhouette_scores_Kmeans))]

plt.figure(figsize=(12, 6))
plt.plot(ward_scores_df['Number of Clusters'], ward_scores_df['Silhouette Score'], label='Ward Linkage')
plt.plot(complete_scores_df['Number of Clusters'], complete_scores_df['Silhouette Score'], label='Complete Linkage')
plt.plot(average_scores_df['Number of Clusters'], average_scores_df['Silhouette Score'], label='Average Linkage')
#plot kmeans
plt.plot(range_n_clusters, silhouette_scores_Kmeans, label='k-means')
plt.title('Silhouette Scores for Different Methods')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.legend()
plt.show()

# ----------------------------------------

# ----------------------------------------
