# Xander Chapman
# CDS1020 HW1

#Imports
import pandas as pd
import matplotlib.pyplot as plt
import scipy as scipy

# Q2 #

list1 = [1,2,3,4,5]
# A list is a single dimensional ordered list of values. 

tuple1 = (1,2,3,4,5)
# A tuple stores multiple items in a single variable. 
tuple2 = (6,7,8,9,10)
tuple3 = tuple1 + tuple2
print(tuple3) ##(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

# A dictonary is an unordered set of information indexed by key. 
d = {}
d[(0,0)] = 3
print(d[(0,0)]) ##3
print(d) ##{(0, 0): 3}

# Q3 #

#tuple1[1] = 500 #TypeError: 'tuple' object does not support item assignment

# Q4 #

var1 = 100
var2 = 5
prod1 = (var1 ** var2)/var2
print(prod1) ##2000000000.0

# Q5 #

data = pd.read_csv("diabetes.csv", header=0)
print(data.head(0)) ##Columns: [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome]

# Q6 #

print(data) ##[768 rows x 9 columns] #Rows start at 0 to 767, Columns are labeled, Everything is numerical
print(data.shape) ##(768, 9) 

# Q7 #
print(data.describe()) #Gives the description of all columns of dataset

# Q8 #

filtered_data = data[(data['Age'] >= 20) & (data['Age'] <= 50)]
print(filtered_data['BMI'].median()) ##32.3
filtered_data['BMI'].plot(kind="box")
plt.show()

# Q9 #

data['Pregnancies'].plot(kind="hist")
plt.show()

# Q10 #
scatdata = data[(data['BloodPressure'] >0) & (data['BMI'] >0)]
scatdata.plot(kind='scatter', x="BloodPressure", y="BMI")
plt.show()

print(scipy.stats.pearsonr(scatdata['BloodPressure'], data['BMI'])) # ~ 0.29 = Blood Pressure and BMI are slightly correlated. 