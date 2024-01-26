# List of required packages
required_packages <- c("caret", "glmnet", "rpart", "rpart.plot", "ROCR", "PRROC")

# Check if packages are installed
new_packages <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]

# Install missing packages
if (length(new_packages) > 0) {
  install.packages(new_packages, dependencies = TRUE)
}


# Load necessary libraries
library(caret)
library(glmnet)
library(rpart)
library(rpart.plot)
library(ROCR)
library(PRROC)

# Load data
data <- read.csv("posts/Problem Set 4/donors.csv")

# Handle missing data
# For simplicity, we'll impute with median for numeric and mode for factors
for (col in names(data)) {
  if (is.numeric(data[, col])) {
    data[is.na(data[, col]), col] <- median(data[, col], na.rm = TRUE)
  } else {
    levels <- unique(data[, col])
    data[is.na(data[, col]), col] <- levels[which.max(tabulate(match(data[, col], levels)))]
  }
}

# Partition data
set.seed(123)
train_index <- createDataPartition(data$respondedMailing, p = 0.75, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Logistic LASSO Model
x <- model.matrix(respondedMailing ~ ., data = train_data)[, -1]
y <- train_data$respondedMailing
cv_lasso <- cv.glmnet(x, y, family = "binomial")
coef(cv_lasso)

# Decision Tree Model
control <- rpart.control(cp = 0, xval = 10) # 10-fold CV
fit_tree <- rpart(respondedMailing ~ ., data = train_data, control = control)
printcp(fit_tree)
rpart.plot(fit_tree)

# Performance on Test Data
# LASSO
# Create model matrix for entire dataset
full_matrix <- model.matrix(respondedMailing ~ ., data = data)[, -1]

# Split this matrix based on previously created train_index
train_matrix <- full_matrix[train_index, ]
test_matrix <- full_matrix[-train_index, ]

# Train the LASSO model using train_matrix
cv_lasso <- cv.glmnet(train_matrix, y, family = "binomial")

# Now, for predictions, use test_matrix
probs_lasso <- predict(cv_lasso, newx = test_matrix, type = "response")


# Decision Tree
# Combine the data
combined_data <- rbind(train_data, test_data)

# Convert state to factor ensuring all levels are present
combined_data$state <- factor(combined_data$state)

# Split the data again
train_data <- combined_data[1:nrow(train_data), ]
test_data <- combined_data[(nrow(train_data) + 1):nrow(combined_data), ]

# Retrain the Decision Tree Model using the updated train_data
fit_tree <- rpart(respondedMailing ~ ., data = train_data, control = control)

# Now try predicting again
predictions_tree <- predict(fit_tree, newdata = test_data, type = "vector")

confusion_tree <- table(test_data$respondedMailing, predictions_tree)
print(confusion_tree)

# ROC plot
# Ensure the target variable is a factor
train_data$respondedMailing <- as.factor(train_data$respondedMailing)
test_data$respondedMailing <- as.factor(test_data$respondedMailing)

# Retrain the Decision Tree Model
fit_tree <- rpart(respondedMailing ~ ., data = train_data, control = control)

# Predict probabilities for the LASSO model
pred_lasso <- prediction(probs_lasso, test_data$respondedMailing)
perf_lasso <- performance(pred_lasso, "tpr", "fpr")

# Predict probabilities for the Decision Tree model
probs_tree <- predict(fit_tree, newdata = test_data, type = "prob")[, 2]
pred_tree <- prediction(probs_tree, test_data$respondedMailing)
perf_tree <- performance(pred_tree, "tpr", "fpr")

# Plot ROC curves
plot(perf_lasso, col = "red")
plot(perf_tree, col = "blue", add = TRUE)
legend("bottomright", legend = c("LASSO", "Decision Tree"), col = c("red", "blue"), lty = 1)


# Precision Recall Chart and Cumulative Gain Chart
# We can use PRROC package for this.

# For LASSO
pr_lasso <- pr.curve(scores.class0 = probs_lasso, weights.class0 = test_data$respondedMailing == "FALSE")

# For Decision Tree
probs_tree <- predict(fit_tree, newdata = test_data, type = "prob")[, 2]
pr_tree <- pr.curve(scores.class0 = probs_tree, weights.class0 = test_data$respondedMailing == "FALSE")

# Compare the two models by AUC
cat("AUC for LASSO:", pr_lasso$auc.integral, "\n")
cat("AUC for Decision Tree:", pr_tree$auc.integral, "\n")
