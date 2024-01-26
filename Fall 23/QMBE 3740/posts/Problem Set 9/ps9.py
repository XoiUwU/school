import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Exercise 1: Clustering Colleges in Indiana Based on Faculty Salary and Tuition

# Load the dataset
file_path = 'posts/Problem Set 9/college.csv'
college_data = pd.read_csv(file_path)

# Filter for colleges in Indiana
# This step focuses on a specific geographical area (Indiana) as per the exercise requirement.
indiana_colleges = college_data[college_data['state'] == 'IN']
indiana_colleges_clustering = indiana_colleges[['tuition', 'faculty_salary_avg']]

# Exercise 2: Selecting Optimal Values for k

# Elbow Method and Silhouette Score
# These methods are used to determine the optimal number of clusters.
k_values = range(2, 10)
sum_of_squared_distances = []
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(indiana_colleges_clustering)
    sum_of_squared_distances.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(indiana_colleges_clustering, kmeans.labels_))

# Plotting the Elbow Method and Silhouette Score
# These plots help in visually identifying the optimal k.
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(k_values, sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method For Optimal k')
plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_scores, 'ro-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score For Different k')
plt.tight_layout()
plt.show()

# Reasoning for k selection:
# k=4 is selected based on the Elbow Method, and k=2 is chosen due to the highest Silhouette Score.

# Exercise 3: Generating Cluster Diagrams

# Function to perform KMeans clustering and plot the results
def plot_clusters(data, k):
    kmeans = KMeans(n_clusters=k, random_state=0)
    data['cluster'] = kmeans.fit_predict(data[['tuition', 'faculty_salary_avg']])
    plt.scatter(data['tuition'], data['faculty_salary_avg'], c=data['cluster'], cmap='viridis', marker='o', edgecolor='black')
    plt.title(f'Cluster of Indiana Colleges with k={k}')
    plt.xlabel('Annual Tuition ($)')
    plt.ylabel('Average Faculty Salary ($)')

# Plotting clusters for k=2 and k=4
# These plots allow for a visual comparison between the two selected k values.
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plot_clusters(indiana_colleges_clustering.copy(), 2)
plt.subplot(1, 2, 2)
plot_clusters(indiana_colleges_clustering.copy(), 4)
plt.tight_layout()
plt.show()

# Justification for the best result:
# The choice between k=2 and k=4 depends on the desired granularity of the clustering.
# k=2 provides a clear and broad categorization, suitable for general analysis.
# k=4 offers a more detailed view, which can be beneficial for in-depth or specific analyses.

























import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

# --- Exercise 4: Determining the Number of Clusters ---

# Load the dataset
file_path = 'posts/Problem Set 9/Cereals.csv'
cereals_df = pd.read_csv(file_path)

# Dropping missing values
cereals_df.dropna(inplace=True)

# Selecting specific columns for clustering (excluding name, mfr, type, weight, shelf, cups, rating)
columns_for_clustering = cereals_df.columns.difference(['name', 'mfr', 'type', 'weight', 'shelf', 'cups', 'rating'])
cereals_subset = cereals_df[columns_for_clustering]

# Elbow Method to determine the optimal number of clusters
def calculate_wcss(data):
    wcss = []
    for n in range(1, 11):
        kmeans = KMeans(n_clusters=n, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    return wcss

wcss = calculate_wcss(cereals_subset)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.show()

# Silhouette Score to assess the quality of clusters
def calculate_silhouette_scores(data):
    silhouette_scores = []
    for n in range(2, 11):
        kmeans = KMeans(n_clusters=n, init='k-means++', max_iter=300, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    return silhouette_scores

silhouette_scores = calculate_silhouette_scores(cereals_subset)

plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='--')
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.xticks(range(2, 11))
plt.show()

# Based on Elbow Method and Silhouette Score, choosing 3 clusters
num_clusters = 3

# --- Exercise 5: Performing k-means Clustering ---

# Performing k-means clustering
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
cereals_df['cluster'] = kmeans.fit_predict(cereals_subset)

# Displaying cluster centers
cluster_centers = kmeans.cluster_centers_
cluster_centers_df = pd.DataFrame(cluster_centers, columns=columns_for_clustering)

# Visualization of cluster centers in the sugars-calories space
plot_data = cereals_df.copy()
plot_data['sugars_scaled'] = (plot_data['sugars'] - plot_data['sugars'].mean()) / plot_data['sugars'].std()
plot_data['calories_scaled'] = (plot_data['calories'] - plot_data['calories'].mean()) / plot_data['calories'].std()

cluster_centers_plot = pd.DataFrame({
    'sugars_scaled': (cluster_centers_df['sugars'] - cereals_df['sugars'].mean()) / cereals_df['sugars'].std(),
    'calories_scaled': (cluster_centers_df['calories'] - cereals_df['calories'].mean()) / cereals_df['calories'].std(),
    'cluster': ['Center 1', 'Center 2', 'Center 3']
})

plt.figure(figsize=(12, 8))
sns.scatterplot(data=plot_data, x='sugars_scaled', y='calories_scaled', hue='cluster', palette='Set1')
sns.scatterplot(data=cluster_centers_plot, x='sugars_scaled', y='calories_scaled', s=200, color='black', marker='X', label='Cluster Centers')
plt.title('Cluster Distribution in Sugars-Calories Space')
plt.xlabel('Scaled Sugars')
plt.ylabel('Scaled Calories')
plt.legend()
plt.show()

# --- Exercise 6: Naming the Clusters ---

# Naming the clusters based on their characteristics
# Cluster 1: Moderate Sugar - Moderate Calorie
# Cluster 2: High Fiber - High Sugar
# Cluster 3: Low Sugar - Low Sodium

