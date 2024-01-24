### Import data. ###
#file = open("MNweather.txt", "r")
#x = []
#y = []
#for line in file:
#    if line:
#        year, temp = line.split()
#        x.append(year)
#        y.append(temp)
#        print(line)
#file.close() #Close the file

### Define a range. ###
#Method 1
0, 1, 2, 3, 4 #data type called range
#Method 2
what_is_this = range(4)
print(type(what_is_this))
#Method 3
for x in range(6):
    print(x)

### Define a dictionary. ###
d = {}
d[(1,2)] = 3
print(d[(1,2)])

h = ["bad","cat","bad",
     "alligator","mom",
     "floatie","pillow"]
d= {}
for word in h:
    if word in d:
        d[word] += 1
    else:
        d[word] = 1
print(d)

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