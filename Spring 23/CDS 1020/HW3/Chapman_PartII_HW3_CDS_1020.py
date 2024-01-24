### Import libraires. ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# *** Insert missing library. *** #
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

### Load data. ###
myData = pd.read_csv(r"HW3/Pima_Indians_diabetes.csv")
myData.shape #(768, 9)
myData.head
type(myData) #pandas.core.frame.DataFrame
myData.iloc[0,:]
myData.head

### Prepare data for splitting. ###
X = myData["Glucose"]
y = myData["Outcome"]
### Split data. ###

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

###Confirm splitting of 80% for training, 20% for testing. ###

print(f'Training set: {len(Xtrain)/len(X)}')
print(f'Testing set: {len(Xtest)/len(X)}')
# *** Include print statements of percentages. *** #

### Create model. ###
clf = LogisticRegression()

#Reshape X training set.
Xtrain = Xtrain.to_numpy(Xtrain)
Xtrain = Xtrain.reshape(614,1)
Xtrain = pd.DataFrame(Xtrain)
print(Xtrain.shape) #(614,1)
print(ytrain.shape) #(614,1)
Xtest = Xtest.to_numpy(Xtest)
Xtest = Xtest.reshape(154,1)
Xtest = pd.DataFrame(Xtest)
print(Xtest.shape) #(154,1)
print(Xtest.shape) #(154,1)

### Fit model to training data. ###
clf.fit(Xtrain, ytrain)

### Predict on testing data. ###
predictions = clf.predict(Xtest)

### Retrieve optimized coefficient. ###
coef = clf.coef_[0][0]
print(f'Coefficient: {coef}')
print('The coefficient represents the change in the log-odds of the outcome for a one unit increase in the feature.')

# *** Insert code here. *** #
# *** Paste its value here. *** #
# *** Interpret the parameter in the context of logistic regression. *** #

### Evaluate performance. ###
confusion_matrix = np.array([[sum((ytest==1) & (predictions==1)),sum((ytest==1) & (predictions==0))],
                           [sum((ytest==0) & (predictions==1)),sum((ytest==0) & (predictions==0))]])

# *** Interpret the confusion matrix in the context of logistic regression. *** #
print(confusion_matrix)
print('The confusion matrix shows the number of true positives, false positives, false negatives, and true negatives.')
