import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.contrast import WaldTestResults
##
## Loading and preprocessing the dataset
##

# Load the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
data = pd.read_csv(url, names=names)

# Replace missing values '?' with NaN
data = data.replace('?', np.nan)

# Convert some columns to numeric
cols = ['trestbps', 'chol', 'thalach', 'oldpeak']
data[cols] = data[cols].apply(pd.to_numeric)

# Impute missing values with column mean
data = data.fillna(data.mean())

# Convert sex to binary (0 = female, 1 = male)
data['sex'] = pd.get_dummies(data['sex'], drop_first=True)

# Convert categorical variables to binary
cp = pd.get_dummies(data['cp'], prefix='cp')
thal = pd.get_dummies(data['thal'], prefix='thal')
slope = pd.get_dummies(data['slope'], prefix='slope')
ca = pd.get_dummies(data['ca'], prefix='ca')
data = pd.concat([data, cp, thal, slope, ca], axis=1)
data = data.drop(['cp', 'thal', 'slope', 'ca'], axis=1)



# Save preprocessed data to CSV file
data.to_csv('heart_disease_preprocessed.csv', index=False)

##
## Unit 1: Strategy 2: Hypothesis Testing for Interference
##
print("Hypothesis Testing for Interference")
print("-------------------------------------------------------------------")


# Define the predictor and target variables
X = data.drop('target', axis=1)
y = data['target']

# Convert target variables to binary
y = (y > 0).astype(int)

value_counts = y.value_counts()
if len(value_counts) > 2:
    print("Target variable is not binary")
    print(value_counts)
else:
    print("Target variable is binary")
print(X)

print("\n")

# Fit a logistic regression model
if np.any((y < 0) | (y > 1)):
    raise ValueError("Response variable must be between 0 and 1")
model = sm.Logit(y, sm.add_constant(X)).fit()

print("\n\n\n")

# Perform the Wald test for each feature at a 0.05 threshold
significant_features = []
for feature in X.columns:
    wald_test = model.wald_test_terms(feature)
    if wald_test.pvalues.size == 0:
        print(f"{feature} is not significant in predicting heart disease at a 0.05 threshold")
    elif wald_test.pvalues[0] < 0.05:
        significant_features.append(feature)
        print(f"{feature} is significant in predicting heart disease (p-value = {wald_test.pvalues[0]:.3f}) at a 0.05 threshold")
    else:
        print(f"{feature} is not significant in predicting heart disease at a 0.05 threshold")
print(f"\nSignificant features at a 0.05 threshold: {significant_features}\n\n")

#

# Perform the Wald test for each feature at a 0.1 threshold
significant_features = []
for feature in X.columns:
    wald_test = model.wald_test_terms(feature)
    if wald_test.pvalues.size == 0:
        print(f"{feature} is not significant in predicting heart disease at a 0.1 threshold")
    elif wald_test.pvalues[0] < 0.1:
        significant_features.append(feature)
        print(f"{feature} is significant in predicting heart disease (p-value = {wald_test.pvalues[0]:.3f}) at a 0.1 threshold")
    else:
        print(f"{feature} is not significant in predicting heart disease at a 0.1 threshold")
print(f"\nSignificant features at a 0.1 threshold: {significant_features}\n\n")

#

# Perform the Wald test for each feature at a 0.25 threshold
significant_features = []
for feature in X.columns:
    wald_test = model.wald_test_terms(feature)
    if wald_test.pvalues.size == 0:
        print(f"{feature} is not significant in predicting heart disease at a 0.25 threshold")
    elif wald_test.pvalues[0] < 0.25:
        significant_features.append(feature)
        print(f"{feature} is significant in predicting heart disease (p-value = {wald_test.pvalues[0]:.3f}) at a 0.25 threshold")
    else:
        print(f"{feature} is not significant in predicting heart disease at a 0.25 threshold")
print(f"\nSignificant features at a 0.25 threshold: {significant_features}\n\n")
