#!/usr/bin/env python
# coding: utf-8

# In[13]:


### Import modules. ###
import numpy as np
from HW3_CDS_1020 import GaussianDiscriminant_Dep, GaussianDiscriminant_Ind

### Load the data. ###
dftrain = np.genfromtxt('HW3/alt_training_data_HW3_CDS_1020.txt')
dftest = np.genfromtxt('HW3/alt_testing_data_HW3_CDS_1020.txt')
Xtrain = dftrain[:,0:8]
ytrain = dftrain[:,8]
Xtest = dftest[:,0:8]
ytest = dftest[:,8]

### Create the multivariate Gaussian classifier that uses class-dependent covariance matrices. ### 
clf_dep = GaussianDiscriminant_Dep(2,8,[0.2,0.8])

### Update the model based on the training data. ###
clf_dep.fit(Xtrain,ytrain)

### Evaluate based on the testing data. ###
predictions = clf_dep.predict(Xtest)
confusion_matrix = np.array([[sum((ytest==1) & (predictions==1)),sum((ytest==2) & (predictions==1))],
                           [sum((ytest==1) & (predictions==2)),sum((ytest==2) & (predictions==2))]])
print('Confusion Matrix for MVN classifier with class-dependent covariance')
print(confusion_matrix)

### Create a multivariate Gaussian classifier that uses class-independent covariance matrices. ###
clf_ind = GaussianDiscriminant_Ind(2,8,[0.2,0.8])

### Update the model based on the training data. ###
clf_ind.fit(Xtrain,ytrain)

### Evaluate based on the testing data. ###
predictions = clf_ind.predict(Xtest)
confusion_matrix = np.array([[sum((ytest==1) & (predictions==1)),sum((ytest==2) & (predictions==1))],
                           [sum((ytest==1) & (predictions==2)),sum((ytest==2) & (predictions==2))]])
print('Confusion Matrix for MVN classifier with class-independent covariance')
print(confusion_matrix)


mean1_dep, mean2_dep, cov1_dep, cov2_dep = clf_dep.get_parameters()
mean1_ind, mean2_ind, cov_ind = clf_ind.get_parameters()


print(f'Mean1 Dep:{mean1_dep}Ind:{mean1_ind}')
print(f'Mean2 Dep:{mean2_dep}Ind:{mean2_ind}')
print(f'Cov dep1:{cov1_dep}dep2:{cov2_dep}ind{cov_ind}')

# In[ ]:




