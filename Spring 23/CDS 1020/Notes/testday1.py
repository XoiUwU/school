### Import Libraries. ###

import csv
import numpy as np
from pprint import pprint

# Download digits dataset. #

with open('Digits089.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

### Load data. ###

Y= np.array([entry[1] for entry in data]).astype(int)
X = np.array([entry[2:] for entry in data]) .astype(float)

print(type(X)) #class 'numpy.ndarray'
print(X.shape) #(3000, 784)
print(Y.shape)