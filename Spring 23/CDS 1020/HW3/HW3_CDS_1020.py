#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### Import modules. ###
import numpy as np


# In[1]:


### Define the class for a Multivariate Gaussian Class-dependent Model. ###
class GaussianDiscriminant_Dep: 
    def __init__(self,k,d,priors=None):
        self.k = k 
        self.d = d
        self.mean = np.zeros((k,d)) #create attribute mean
        self.S = np.zeros((k,d,d)) #create attribute covariance matrix
        if priors is not None: 
            self.pi = priors #create attribute prior, pi (if there is a prior)
        else: 
            self.pi = [1.0/k for i in range(k)] #assumes the priors are equal, if not given
        
    def fit(self, Xtrain, ytrain): 
        
        for i in range(self.k): #calculate the mean for each class
            self.mean[i,:]=np.mean((Xtrain[ytrain==i+1]),axis=0)
        
        for i in range(self.k): #calculate the class-dependent covariance matrix
            for i in range(self.k):
                self.S[i,:,:]=np.cov(np.transpose(Xtrain[ytrain==i+1]))
            
    def predict(self, Xtest): #define predict function to predict a y using the test set of Xs
        predLabel = np.ones(Xtest.shape[0]) #create data structure (i.e., vector) the length
        #of the number of X variables in the testing set. This data structure is filled with ones.
        
        for element in np.arange(Xtest.shape[0]): #for each test set example, we will derive best value and prediction.
            opt_val = -float('inf')
            opt_pred = 0
            for c in np.arange(self.k): #where we calculate the discriminant function value for each class
                my_val = -0.5*np.log(np.linalg.det(self.S[c,:,:]))-0.5*np.matmul(np.matmul(np.transpose(Xtest[element,:]-self.mean[c,:]),np.linalg.inv(self.S[c,:,:])),(Xtest[element,:]-self.mean[c,:]))+np.log(self.pi[c])
                if my_val > opt_val:
                    opt_val = my_val #assign the best value
                    opt_pred = c #assign the optimal class
            predLabel[element] = opt_pred+1 #necessary as our class labels are 1 and 2
        
        return predLabel
    
    def get_parameters(self):
        return self.mean[0],self.mean[1],self.S[0,:,:],self.S[1,:,:]

        


# In[2]:


### Define the class for the Multivariate Gaussian Class-independent Model. ###
class GaussianDiscriminant_Ind: 
    def __init__(self,k,d,priors=None):
        self.k = k 
        self.d = d
        self.mean = np.zeros((k,d)) #create attribute mean
        self.S = np.zeros((d,d)) #create attribute covariance matrix
        if priors is not None: 
            self.pi = priors #create attribute prior, pi (if there is a prior)
        else: 
            self.pi = [1.0/k for i in range(k)] #assumes the priors are equal, if not given
        
    def fit(self, Xtrain, ytrain): 
        
        for i in range(self.k): #calculate the mean for each class.
            self.mean[i,:]=np.mean((Xtrain[ytrain==i+1]),axis=0)
        
        group_covar_mats = []
        for i in range(self.k): #calculate the class-dependent covariance matrix
            group_covar_mats += [np.cov(np.transpose(Xtrain[ytrain==1+1]))]
        my_covar = sum([group_covar_mats[i]*self.pi[i] for i in range(self.k)])
        self.S[:,:] = my_covar
            
    def predict(self, Xtest): #define predict function to predict a y using the test set of Xs 
        predLabel = np.ones(Xtest.shape[0]) #create data structure (i.e., vector) the length
        #of the number of X variables in the testing set. This data structure is filled with ones.
        
        for element in np.arange(Xtest.shape[0]): #for each test set example, we will derive best value and prediction.
            opt_val = -float('inf')
            opt_pred = 0
            for c in np.arange(self.k): #Where we calculate the discriminant function value for each class
                my_val = -0.5*np.log(np.linalg.det(self.S[:,:]))-0.5*np.matmul(np.matmul(np.transpose(Xtest[element]-self.mean[c,:]),np.linalg.inv(self.S[:,:])),(Xtest[element]-self.mean[c,:]))+np.log(self.pi[c])
                if my_val > opt_val:
                    opt_val = my_val #assigns the best value
                    opt_pred = c #assigns the optimal class
            predLabel[element] = opt_pred+1 #necessary as our class labels are 1 and 2
        
        return predLabel
    
    def get_parameters(self):
        return self.mean[0],self.mean[1],self.S


# In[ ]:




