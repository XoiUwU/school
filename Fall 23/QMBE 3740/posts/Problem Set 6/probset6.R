# Load libraries
library(tidyverse)
library(dplyr)
library(caret)
library(rpart)
library(randomForest)
library(h2o)
library(pROC)
library(ggplot2)

# Load data
data <- read_csv("posts/Problem Set 6/UniversalBank.csv")

# 1. Partition the data
x <- data %>% select(-PersonalLoan)
y <- data$PersonalLoan
data$PersonalLoan <- as.factor(data$PersonalLoan)
train_index <- createDataPartition(data$PersonalLoan, p = .8, list = FALSE, times = 1)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# 2. Build the best tuned decision tree
tune_control <- rpart.control(cp = seq(0.001, 0.1, by = 0.001))
dt_fit <- rpart(PersonalLoan ~ ., data = train_data, method = "class", control = tune_control)
pred_dt <- as.factor(ifelse(predict(dt_fit, test_data, type = "prob")[, 2] > 0.5, 1, 0))

# 3. Build the best tuned random forest
rf_fit <- randomForest(PersonalLoan ~ ., data = train_data, ntree = 100)
pred_rf <- as.factor(ifelse(predict(rf_fit, test_data, type = "prob")[, 2] > 0.5, 1, 0))

# 4. Build the best tuned gradient boosting machine
lgb_train <- lgb.Dataset(data = as.matrix(train_data %>% select(-PersonalLoan)), label = train_data$PersonalLoan)
params <- list(
  objective = "binary",
  metric = "binary_error",
  boost_from_average = FALSE
)
lgb_fit <- lgb.train(
  params = params,
  data = lgb_train,
  nrounds = 100,
  valids = list(
    train = lgb_train
  ),
  early_stopping_rounds = 10,
  eval_freq = 10
)
prob_lgb <- predict(lgb_fit, as.matrix(test_data %>% select(-PersonalLoan)))  # Adjusted to prob_lgb and used as.matrix
pred_lgb <- as.factor(ifelse(prob_lgb > 0.5, 1, 0))

# 5. Compare the precision and sensitivity of all three models on the testing data
cm_dt <- confusionMatrix(pred_dt, test_data$PersonalLoan)
cm_rf <- confusionMatrix(pred_rf, test_data$PersonalLoan)
cm_lgb <- confusionMatrix(as.factor(ifelse(prob_lgb > 0.5, 1, 0)), test_data$PersonalLoan)
precision_dt <- cm_dt$byClass['Pos Pred Value']
sensitivity_dt <- cm_dt$byClass['Sensitivity']
precision_rf <- cm_rf$byClass['Pos Pred Value']
sensitivity_rf <- cm_rf$byClass['Sensitivity']
precision_lgb <- cm_lgb$byClass['Pos Pred Value']
sensitivity_lgb <- cm_lgb$byClass['Sensitivity']
comparison_df <- data.frame(
  Model = c("Decision Tree", "Random Forest", "LightGBM"),
  Precision = c(precision_dt, precision_rf, precision_lgb),
  Sensitivity = c(sensitivity_dt, sensitivity_rf, sensitivity_lgb)
)
print(comparison_df)

# 6. Create an ROC plot comparing all three models on the testing data. Which has the greatest AUC?
y_test <- test_data$PersonalLoan
roc_dt <- roc(y_test, as.numeric(pred_dt))
roc_rf <- roc(y_test, as.numeric(pred_rf))
roc_lgb <- roc(y_test, as.numeric(pred_lgb))
roc_list <- list(
  DecisionTree = roc_dt,
  RandomForest = roc_rf,
  LightGBM = roc_lgb
)
roc_multi_plot <- ggroc(roc_list)
print(roc_multi_plot)
print(paste("Decision Tree ROC", auc(roc_dt)))
print(paste("Random Forect ROC", auc(roc_rf)))
print(paste("Gradient Boosting ROC", auc(roc_lgb)))
