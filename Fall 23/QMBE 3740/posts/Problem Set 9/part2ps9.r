# Load necessary libraries
library(tidyverse)
library(cluster)
library(factoextra)
library(ggplot2)

# --- Exercise 4: Determining the Number of Clusters ---

# Load the dataset
file_path <- "posts/Problem Set 9/Cereals.csv"
cereals_df <- read.csv(file_path, stringsAsFactors = FALSE)

# Dropping missing values
cereals_df <- na.omit(cereals_df)

# Selecting specific columns for clustering (excluding name, mfr, type, weight, shelf, cups, rating)
columns_for_clustering <- setdiff(names(cereals_df), c("name", "mfr", "type", "weight", "shelf", "cups", "rating"))
cereals_subset <- cereals_df[columns_for_clustering]

# Elbow Method to determine the optimal number of clusters
calculate_wcss <- function(data) {
  wcss <- numeric(10)
  for (n in 1:10) {
    set.seed(42)
    kmeans_result <- kmeans(data, centers = n, nstart = 10)
    wcss[n] <- kmeans_result$tot.withinss
  }
  return(wcss)
}

wcss <- calculate_wcss(cereals_subset)

ggplot() +
  geom_line(aes(x = 1:10, y = wcss), color = "blue") +
  geom_point(aes(x = 1:10, y = wcss), color = "red") +
  labs(title = "Elbow Method", x = "Number of clusters", y = "WCSS") +
  theme_minimal()

# Silhouette Score to assess the quality of clusters
calculate_silhouette_scores <- function(data) {
  silhouette_scores <- numeric(9)
  for (n in 2:10) {
    set.seed(42)
    kmeans_result <- kmeans(data, centers = n, nstart = 10)
    silhouette_avg <- mean(silhouette(kmeans_result$cluster, dist(data))[, "sil_width"])
    silhouette_scores[n - 1] <- silhouette_avg
  }
  return(silhouette_scores)
}

silhouette_scores <- calculate_silhouette_scores(cereals_subset)

ggplot() +
  geom_line(aes(x = 2:10, y = silhouette_scores), color = "blue") +
  geom_point(aes(x = 2:10, y = silhouette_scores), color = "red") +
  labs(title = "Silhouette Scores for Different Numbers of Clusters", x = "Number of Clusters", y = "Silhouette Score") +
  theme_minimal()

# Based on Elbow Method and Silhouette Score, choosing 3 clusters
num_clusters <- 3

# --- Exercise 5: Performing k-means Clustering ---

# Performing k-means clustering
set.seed(42)
kmeans <- kmeans(cereals_subset, centers = num_clusters, nstart = 10)
cereals_df$cluster <- as.factor(kmeans$cluster)

# Displaying cluster centers
cluster_centers_df <- as.data.frame(kmeans$centers)

# Visualization of cluster centers in the sugars-calories space
plot_data <- cereals_df %>%
  mutate(sugars_scaled = scale(sugars),
         calories_scaled = scale(calories))

cluster_centers_plot <- data.frame(
  sugars_scaled = scale(cluster_centers_df$sugars, center = mean(cereals_df$sugars), scale = sd(cereals_df$sugars)),
  calories_scaled = scale(cluster_centers_df$calories, center = mean(cereals_df$calories), scale = sd(cereals_df$calories)),
  cluster = c("Center 1", "Center 2", "Center 3")
)

ggplot(plot_data, aes(x = sugars_scaled, y = calories_scaled, color = cluster)) +
  geom_point() +
  geom_point(data = cluster_centers_plot, aes(x = sugars_scaled, y = calories_scaled), color = "black", size = 4, shape = 4) +
  labs(title = "Cluster Distribution in Sugars-Calories Space", x = "Scaled Sugars", y = "Scaled Calories") +
  theme_minimal()

# --- Exercise 6: Naming the Clusters ---

# Naming the clusters based on their characteristics
# Cluster 1: Moderate Sugar - Moderate Calorie
# Cluster 2: High Fiber - High Sugar
# Cluster 3: Low Sugar - Low Sodium