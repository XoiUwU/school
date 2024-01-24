# Imports!
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, classification_report, multilabel_confusion_matrix
from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Remove futureproofing warnings to reduce clutter
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

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

print("\n\n\n")



##
## Unit 2: Strategy 1: Classification using Scikit-Learn
##
print("\n\n\nClassification using Scikit-Learn")
print("----------------------------------------------------------------")


# Load the data
data = pd.read_csv('heart_disease_preprocessed.csv')

# Split the data into input features and target variable
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# LDA
#Train linear discriminant analysis model on training set
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Evaluate performance on testing set
accuracy = lda.score(X_test, y_test)
print(f"LDA Accuracy: {accuracy}")


# Linear Regression
# Train linear regression model on training set
lr = LinearRegression()
lr.fit(X_train, y_train)

# Evaluate performance on testing set
y_pred = lr.predict(X_test)
accuracy = lr.score(X_test, y_test)
print(f"Linear Regression Accuracy: {accuracy}")


# Logistic Regression
# Define parameter grid
param_grid = {'C': [0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}

# Create logistic regression model
lr = LogisticRegression()

# Create grid search object
grid_search = GridSearchCV(lr, param_grid=param_grid, cv=5)

# Fit grid search to data
grid_search.fit(X_train, y_train)

# Print best parameters and accuracy score
print("Best Parameters: ", grid_search.best_params_)
print("Accuracy Score: ", grid_search.best_score_)

# Train logistic regression model on training set
lr.fit(X_train, y_train)

# Predict on test set
y_pred = lr.predict(X_test)

# Print classification report
report = classification_report(y_test, y_pred, zero_division=1)
print(f"Logistic Regression Classification report:\n{report}")

# Create logistic regression model
clf = LogisticRegression(max_iter=100000)
clf.fit(X_train, y_train)

# Predict on test set
y_pred = clf.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {accuracy}")

# Compute the confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
print(f"Logistic Regression Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# Generate the classification report
report = classification_report(y_test, y_pred)
print(f"Logistic Regression Classification report:\n{report}")


# Ridge Regression
# Train Ridge regression model on training set with alpha=1
ridge = Ridge(alpha=1)
ridge.fit(X_train, y_train)

# Evaluate performance on testing set
y_pred = ridge.predict(X_test)
accuracy = ridge.score(X_test, y_test)
print(f"Ridge Regression Accuracy: {accuracy}")

# K-Nearest Neighbors
# Train KNN model on training set with k=25
knn = KNeighborsClassifier(n_neighbors=25)
knn.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = knn.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Accuracy: {accuracy}")

# Compute the confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
print(f"KNN Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")


# Decision Tree
# Train decision tree model on training set
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = dt.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {accuracy}")

# Compute the confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
print(f"Decision Tree Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# Evaluate performance on testing set
mse = mean_squared_error(y_test, y_pred)
print(f"Decision Tree Mean squared error: {mse}")

# Random Forest
# Train random forest model on training set
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rf.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy}")

# Compute the confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
print(f"Random Forest Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# Evaluate performance on testing set
mse = mean_squared_error(y_test, y_pred)
print(f"Random Forest Mean squared error: {mse}")

# AdaBoost
# Train AdaBoost model on training set
adaboost = AdaBoostClassifier()
adaboost.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = adaboost.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"AdaBoost Accuracy: {accuracy}")

# Compute the confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
print(f"AdaBoost Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# Evaluate performance on testing set
mse = mean_squared_error(y_test, y_pred)
print(f"AdaBoost Mean squared error: {mse}")
