# Load required libraries
library(tidyverse)
library(randomForest)
library(Metrics)
library(caret)

# Load the datasets
pre_crisis_data <- read.csv("posts/Problem Set 7/PreCrisisCV.csv")
post_crisis_data <- read.csv("posts/Problem Set 7/PostCrisisCV.csv")
test_data <- read.csv("posts/Problem Set 7/OnMarketTest-1.csv")

# Split the pre-crisis data into training and validation sets
set.seed(42)
sample_index_pre <- sample(seq_len(nrow(pre_crisis_data)), 0.8 * nrow(pre_crisis_data))
x_train_pre <- pre_crisis_data[sample_index_pre, !(colnames(pre_crisis_data) %in% c("Price", "Property"))]
y_train_pre <- pre_crisis_data[sample_index_pre, "Price"]
x_val_pre <- pre_crisis_data[-sample_index_pre, !(colnames(pre_crisis_data) %in% c("Price", "Property"))]
y_val_pre <- pre_crisis_data[-sample_index_pre, "Price"]

# Train a RandomForest regressor on pre-crisis data
model_pre <- randomForest(x_train_pre, y_train_pre, ntree=50, mtry=2, importance=TRUE)
y_pred_pre <- predict(model_pre, x_val_pre)
mae_pre <- mae(y_val_pre, y_pred_pre)
mse_pre <- mse(y_val_pre, y_pred_pre)
r2_pre <- postResample(y_pred_pre, y_val_pre)[3]

# Split the post-crisis data into training and validation sets
sample_index_post <- sample(seq_len(nrow(post_crisis_data)), 0.8 * nrow(post_crisis_data))
x_train_post <- post_crisis_data[sample_index_post, !(colnames(post_crisis_data) %in% c("Price", "Property"))]
y_train_post <- post_crisis_data[sample_index_post, "Price"]
x_val_post <- post_crisis_data[-sample_index_post, !(colnames(post_crisis_data) %in% c("Price", "Property"))]
y_val_post <- post_crisis_data[-sample_index_post, "Price"]

# Train a RandomForest regressor on post-crisis data
model_post <- randomForest(x_train_post, y_train_post, ntree=50, mtry=2, importance=TRUE)
y_pred_post <- predict(model_post, x_val_post)
mae_post <- mae(y_val_post, y_pred_post)
mse_post <- mse(y_val_post, y_pred_post)
r2_post <- postResample(y_pred_post, y_val_post)[3]

cat("Pre-crisis Metrics:", mae_pre, mse_pre, r2_pre, "\n")
cat("Post-crisis Metrics:", mae_post, mse_post, r2_post, "\n")

# Predict on the test set using the post-crisis model
x_test <- test_data[, !(colnames(test_data) %in% c("Price", "Property"))]
predicted_prices_post <- predict(model_post, x_test)

# Descriptive statistics for the predicted prices using the post-crisis model
predicted_prices_summary_post <- summary(predicted_prices_post)
print(predicted_prices_summary_post)
