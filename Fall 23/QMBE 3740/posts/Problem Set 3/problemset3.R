library(ggplot2)
library(rpart)
library(rpart.plot)

# Read the data
data <- read.csv("posts/Problem Set 3/ToyotaCorolla.csv", header = TRUE)
head(data)

# Check the structure of the dataset
str(data)

# Summary statistics for the dataset
summary(data)

# Check for missing values
missing_values <- sapply(data, function(x) sum(is.na(x)))
missing_values

# Histogram for Price
hist(data$Price, breaks = 50, main = "Price Distribution", xlab = "Price")

# QQ-plot for Price
qqnorm(data$Price)
qqline(data$Price)

# Scatter plot for Age vs. Price
ggplot(data, aes(x = Age_08_04, y = Price)) +
  geom_point() +
  geom_smooth(method = "lm") +
  labs(title = "Price vs Age_08_04")

# Scatterplot: Price vs KM
ggplot(data, aes(x = KM, y = Price)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess", span = 0.75) +
  labs(title = "Price vs KM", x = "KM", y = "Price")

# Boxplot: Price vs Fuel_Type
ggplot(data, aes(x = Fuel_Type, y = Price)) +
  geom_boxplot() +
  labs(title = "Price Distribution by Fuel Type", x = "Fuel Type", y = "Price")

# Scatterplot: Price vs Age, colored by Fuel_Type
ggplot(data, aes(x = Age_08_04, y = Price, color = Fuel_Type)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm") +
  labs(title = "Price vs Age, Colored by Fuel Type", x = "Age", y = "Price")

# Remove columns with zero variance
zero_var_cols <- sapply(data, function(x) if (is.numeric(x)) var(x, na.rm = TRUE) == 0 else FALSE)
data <- data[, !zero_var_cols]

# Splitting data
set.seed(123)
train_indices <- sample(seq_len(nrow(data)), 0.7 * nrow(data))
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

# Fit the regression tree model
fit <- rpart(Price ~ ., data = train_data)

# Prune the tree
pruned_fit <- rpart::prune(fit, cp = 0.1)

# Display the pruned tree using prp without verbose leaf labels
rpart.plot(pruned_fit,
           cex = 0.7,
           main = "Pruned Regression Tree",
           uniform = TRUE,
           branch = 0.5)


library(iml)

predictor <- Predictor$new(pruned_fit, data = train_data, y = train_data$Price)
importance <- FeatureImp$new(predictor, loss = "rmse", compare = "ratio")
importance$plot()

# Using the previous results from the iml
feature_importances <- importance$results

str(feature_importances)

# Sorting the features by importance
sorted_features <- feature_importances[order(feature_importances$Importance), ]

# Selecting the two least important features
least_important_features <- head(sorted_features$Feature, 5)

least_important_features

# Removing less important features
train_data_simplified <- train_data[, !names(train_data) %in% c(least_important_features)]
test_data_simplified <- test_data[, !names(test_data) %in% c(least_important_features)]

# Refit the regression tree model
fit_simplified <- rpart(Price ~ ., data = train_data_simplified)

# Prune the simplified tree
pruned_fit_simplified <- rpart::prune(fit_simplified, cp = 0.1)

# Visualizing the pruned simplified tree
rpart.plot(pruned_fit_simplified, cex = 0.7, main = "Pruned Regression Tree (Simplified)", uniform = TRUE, branch = 0.5)


# Using 10-fold cross-validation
cv_simplified <- rpart::rpart.cv(Price ~ ., data = train_data_simplified, cp = seq(0.01, 0.2, by = 0.01))

# Identifying the optimal cp value
best_cp <- cv_simplified$cp[which.min(cv_simplified$cptable[, "xerror"])]

# Pruning with best cp
final_pruned_simplified <- rpart::prune(fit_simplified, cp = best_cp)


# Predicting on the test set
predictions <- predict(final_pruned_simplified, newdata = test_data_simplified)

# Calculate RMSE on the test set
test_rmse <- sqrt(mean((test_data_simplified$Price - predictions)^2))

# Printing RMSE
test_rmse

# Extracting CV RMSE for the best cp
cv_rmse <- sqrt(min(cv_simplified$cptable[, "xerror"]))

# Printing the errors
cat("Cross-Validation RMSE:", cv_rmse, "\n")
cat("Test RMSE:", test_rmse, "\n")
