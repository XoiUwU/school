library(readr)

# Read the dataset
soap_data <- read_csv("posts/Problem Set 9/BathSoapHousehold.csv")

# Select the relevant columns and scale
soap_scaled <- scale(soap_data[, c("CHILD", "Affluence Index")])

library(factoextra)
set.seed(123) # For reproducibility

p1 <- fviz_nbclust(soap_scaled, kmeans, method = "wss") +
  geom_vline(xintercept = 4, linetype = 2) +
  labs(subtitle = "Elbow method")

p2 <- fviz_nbclust(soap_scaled, kmeans, method = "silhouette") +
  labs(subtitle = "Silhouette method")

set.seed(123) # For reproducibility
k_optimal <- 4 # Replace this with the number of clusters you choose
kmeans_result <- kmeans(soap_scaled, centers = k_optimal, nstart = 25)

p3 <- fviz_cluster(kmeans_result, data = soap_scaled, geom = "point")

soap_data_with_clusters <- soap_data %>%
  mutate(cluster = kmeans_result$cluster)

average_values <- soap_data_with_clusters %>%
  group_by(cluster) %>%
  summarise_at(vars(Value, `Total Volume`), mean)

print(p1)
print(p2)
print(p3)
print(soap_data_with_clusters)
print(average_values)