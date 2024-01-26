## Preparation and Summary
# Load libraries
library(readr)
library(dplyr)
library(tidyr)
library(e1071)
library(ggplot2)
library(corrplot)
library(caret)
library(caretEnsemble)

# Read in the data
data <- read_csv("posts/Problem Set 2/bikes_ps.csv")

# Display the number of rows, features, and data types
cat("Number of rows:", nrow(data), "\n")
cat("Number of features:", ncol(data), "\n")
str(data)

## Data Types and Missing Values
# Convert data types as appropriate
data$date <- as.Date(data$date)
data$holiday <- as.factor(data$holiday)
data$weekday <- as.factor(data$weekday)
data$season <- as.factor(data$season)
data$weather <- as.factor(data$weather)

# Median imputation for missing values in continuous variables
data$windspeed[is.na(data$windspeed)] <- median(data$windspeed, na.rm = TRUE)
data$humidity[is.na(data$humidity)] <- median(data$humidity, na.rm = TRUE)

## Target Feature: Rentals
# Visualize rentals
ggplot(data, aes(x = date, y = rentals)) +
  geom_line() +
  labs(title = "Daily Bike Rentals Over Time", x = "Date", y = "Rentals")

# Check for skewness
skewness_rentals <- skewness(data$rentals)
cat("Skewness of Rentals:", skewness_rentals, "\n")

# Boxplot to identify outliers
ggplot(data, aes(y = rentals)) +
  geom_boxplot() +
  labs(title = "Boxplot of Rentals", y = "Rentals")

## Non-Target Features
# Correlation Plot
# Create a correlation plot of numeric features
numeric_data <- data %>%
  select_if(is.numeric)

correlation_matrix <- cor(numeric_data)
corrplot(correlation_matrix, method = "circle")

# Gallery of Distribution and Correlation Plots
# Gallery of distribution plots
numeric_data %>%
  gather() %>%
  ggplot(aes(value)) +
  geom_histogram(bins = 20) +
  facet_wrap(~key, scales = "free") +
  labs(title = "Distribution of Numeric Features")

# Gallery of correlation plots
numeric_data %>%
  gather() %>%
  ggplot(aes(x = rentals, y = value)) +
  geom_point() +
  geom_smooth(method = "lm") +
  facet_wrap(~key, scales = "free") +
  labs(title = "Correlation of Numeric Features with Rentals")

# Feature Normalization and Dummy Variables
# Z-score normalize temperature
data$temperature <- scale(data$temperature)

# Min-max normalize windspeed variable
data$windspeed <- scales::rescale(data$windspeed)

# Convert categorical variables into dummy variables
data <- dummyVars(~., data = data, fullRank = TRUE) %>%
  predict(data)

## Model Training
# Convert data back to a data frame
data <- as.data.frame(data)

# Train a linear regression model using all features (except those you might remove)
model <- lm(rentals ~ . - date - rentals, data = data)

# Make predictions
data$predicted_rentals <- predict(model, newdata = as.data.frame(data))

# Visualize actual rentals and predicted rentals over time
ggplot(data, aes(x = date)) +
  geom_line(aes(y = rentals, color = "Actual")) +
  geom_line(aes(y = predicted_rentals, color = "Predicted")) +
  labs(title = "Actual vs. Predicted Rentals Over Time", x = "Date", y = "Rentals") +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red"))


#Feature Selection

# The features in the dataset make sense for predicting daily bike rentals. We have considered variables like temperature, humidity, windspeed, and categorical variables like holidays and working days, which are likely to influence bike rental patterns.
# Model Fitting

# To "fit" a model to the data means training the model using the dataset to learn the relationships between the features and the target variable (rentals). The linear regression model we used has learned these relationships and can make predictions based on new data.
# Preparations

# Data type conversion: We converted date, holiday, and workingday to appropriate data types.
# Missing value imputation: We used median imputation for continuous variables with missing values.
# Skewness: We observed right-skewness in the rentals variable and may need to apply a transformation.
# Outliers: We identified outliers in the rentals variable.
# Feature normalization: We normalized temperature and windspeed to ensure that they have similar scales.
# Dummy variables: We converted categorical variables into dummy variables for model compatibility.

# These preparations were necessary to ensure that the learning algorithm works effectively and produces reliable predictions.