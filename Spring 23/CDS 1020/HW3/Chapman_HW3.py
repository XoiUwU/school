### PART 1 ###
### a ###
# Assuming equal priors 
P_C1 = P_C2 = 0.5
x = 8.9
#Only using the numerator of the Bayes' theorem

#P(C1|x=8.9) => P(x=8.9|C1)*P(C1) = (1/5)(.5) = 0.1
P_x_given_C1 = 1/5
P_C1_given_x = P_x_given_C1 * P_C1
#P(C2|x=8.9) => P(x=8.9|C2)*P(C2) = (1/6)(10-8.9)(.5) = 0.0917
P_x_given_C2 = (1/6)*(10-x)
P_C2_given_x = P_x_given_C2 * P_C2
#Since P(C1|x=8.9) > P(C2|x=8.9), we can classify the object as belonging to C1.
if P_C1_given_x > P_C2_given_x:
    print(f'Assuming equal priors and x={x}, the observation belongs to C1')
else:
    print(f'Assuming equal priors and x={x}, the observation belongs to C2')

### b ###
#Assuming unequal priors
P_C1 = 0.8
P_C2 = 0.2
x = 3
#Only using the numberator of the Bayes' theorem
#P(C1|x=3) => P(x=3|C1)*P(C1) = (1/5)(.8) = 0.16
P_C1_given_x = P_x_given_C1 * P_C1
#P(C2|x=3) => P(x=3|C2)*P(C2) = (1/6)(3-1)(0.2) = 0.0667
P_x_given_C2 = (1/6)*(x-1)
P_C2_given_x = P_x_given_C2 * P_C2
# Since P(C1|x=3) > P(C2|x=3), we can classify the object as belonging to C1.
if P_C1_given_x > P_C2_given_x:
    print(f'Assuming equal priors and x={x}, the observation belongs to C1')
else:
    print(f'Assuming equal priors and x={x}, the observation belongs to C2')


### PART 2 ###
# In File: Chapman_PartII_HW3_CDS_1020.py

### PART 3 ###
### 1 ###
#The __init__ function constructs a new object.
#Another name for this function would be a constructor.
#In this function, the parameters are k, d, and priors (optional)
#k represents the number of classes
#d represents the number of features
#priors represents the prior probabilites of each class. If no priors are passed, they are assumed to be equal of all classes.
#they are not hard coded so that you can reuse the same function without having to rewrite the entire thing again
#the variables created are mean, S, and pi
#mean represents the mean of each class
#S represents the covariance matrix of each class
#pi represents the prior probabilites of each class

### 2 ###
import numpy as np
dftrain = np.genfromtxt('HW3/alt_training_data_HW3_CDS_1020.txt')
cov_matrix = np.cov(dftrain)
print(cov_matrix)
size = dftrain.shape
print(f'The size of the matrix is {size}') #(99, 9)
#Each cell in the covariance matrix represents the covariance between the two variables
#If the covariance is positive, when one variable increases, so does the other.
#If the covariance is negative, when one variable increases, the other decreases
#If the covariance is close to zero, there is little/no linear relationship between the two variables.

### 3 ###
#The main point of a discriminant function in parametric generative classification is to find a linear combination 
#of features that characterizes or separates two or more classes of objects.
#LDA uses discriminant function as the final classifier

### 4 ###
#The code uses np.linalg.det and np.linalg.inv
#np.linalg.inv() is used to compute the inerse of the convariance matrix 
#a matrix is passed into the function and the inverse is returned

### 5 ###
#LDA assumes that the covariance matrix is the same for all classes, LDA can only learn linear boundaries
#QDA assumes that each class has its own covariance matrix, QDA can lean quadratic boundaries

### 6 ###
#Confusion Matrix for MVN classifier with class-dependent covariance
#[[57  0]
# [ 0 42]]
#Confusion Matrix for MVN classifier with class-independent covariance
#[[57  0]
# [ 0 42]]

### 7 ###
#np.genfromtxt returns an ndarray

### 8 ###
#The results of the confusion matrices tell us that all 57 objects belonging to the first class were correcly classified
#and all 42 objects belonging to the second class were also correcly classified for both the class-dependent and class-independent
#covariance.

### 9 ###

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
dftrain = np.genfromtxt('HW3/training_data_HW3_CDS_1020.txt')
dftest = np.genfromtxt('HW3/testing_data_HW3_CDS_1020.txt')
Xtrain = dftrain[:,0:8]
ytrain = dftrain[:,8]
Xtest = dftest[:,0:8]
ytest = dftest[:,8]
priors = [0.2,0.8]
#LDA
lda = LinearDiscriminantAnalysis(priors=priors)
lda.fit(Xtrain, ytrain)
y_pred_lda = lda.predict(Xtest)
confusion_matrix_lda = confusion_matrix(ytest, y_pred_lda)

#QDA
qda = QuadraticDiscriminantAnalysis(priors=priors)
qda.fit(Xtrain, ytrain)
y_pred_qda = qda.predict(Xtest)
confusion_matrix_qda = confusion_matrix(ytest, y_pred_qda)

print('LDA confusion matrix:')
print(confusion_matrix_lda)
print('QDA confusion matrix:')
print(confusion_matrix_qda)


### 10 ###
#Comparing the output of the get_parameters() methods we see that the mean vectors have the same dimensions
#The GaussianDiscriminant_Dep class returns a separate covariance matrix for each class
#The GaussianDiscriminant_Ind class returns a single covariance matrix that is shared by both classes
#This is because class-independent analysis assums all classes share the same covariance matrix


### 11 ###
#Confusion Matrix for MVN classifier with class-dependent covariance
#[[ 8  6]
# [49 36]]
#Confusion Matrix for MVN classifier with class-independent covariance
#[[ 3  1]
# [54 41]]
#The class-dependent matrix has 6 false positives and 49 false negatives
#The class-independent matrix has 1 false positives and 54 false negatives
#Comparing to the original testing/training data, this alt data isn't as good because it leads to more false classifications