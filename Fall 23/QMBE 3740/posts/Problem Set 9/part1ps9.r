library(dplyr)
library(ggplot2)
library(factoextra)
library(cluster)

# Exercise 1: Clustering Colleges in Indiana Based on Faculty Salary and Tuition

# Load the dataset
college_data <- read.csv("posts/Problem Set 9/college.csv")

# Filter for colleges in Indiana
# This step focuses on a specific geographical area (Indiana) as per the exercise requirement.
indiana_colleges <- filter(college_data, state == "IN")
indiana_colleges_clustering <- select(indiana_colleges, tuition, faculty_salary_avg)

# Exercise 2: Selecting Optimal Values for k

# Elbow Method and Silhouette Score
# These methods are used to determine the optimal number of clusters.
k_values <- 2:9
sum_of_squared_distances <- numeric()
silhouette_scores <- numeric()

for (k in k_values) {
  set.seed(0)
  kmeans_result <- kmeans(indiana_colleges_clustering, centers = k, nstart = 25)
  sum_of_squared_distances[k - 1] <- kmeans_result$tot.withinss
  silhouette_scores[k - 1] <- mean(silhouette(kmeans_result$cluster, dist(indiana_colleges_clustering))[, 3])
}

# Plotting the Elbow Method and Silhouette Score
# These plots help in visually identifying the optimal k.
plot(k_values, sum_of_squared_distances, type = "b", xlab = "k", ylab = "Sum of Squared Distances", main = "Elbow Method For Optimal k")
plot(k_values, silhouette_scores, type = "b", xlab = "k", ylab = "Silhouette Score", main = "Silhouette Score For Different k")

# Exercise 3: Generating Cluster Diagrams

plot_clusters <- function(data, k) {
  set.seed(0)
  kmeans_result <- kmeans(data[, c("tuition", "faculty_salary_avg")], centers = k, nstart = 25)
  data$cluster <- kmeans_result$cluster
  ggplot(data, aes(x = tuition, y = faculty_salary_avg, color = as.factor(cluster))) + geom_point() +
    ggtitle(paste("Cluster of Indiana Colleges with k=", k)) + xlab("Annual Tuition ($)") + ylab("Average Faculty Salary ($)")
}

# Plotting clusters for k=2 and k=4
# These plots allow for a visual comparison between the two selected k values.
plot_clusters(indiana_colleges_clustering, 2)
plot_clusters(indiana_colleges_clustering, 4)

# Justification for the best result:
# The choice between k=2 and k=4 depends on the desired granularity of the clustering.
# k=2 provides a clear and broad categorization, suitable for general analysis.
# k=4 offers a more detailed view, which can be beneficial for in-depth or specific analyses.
