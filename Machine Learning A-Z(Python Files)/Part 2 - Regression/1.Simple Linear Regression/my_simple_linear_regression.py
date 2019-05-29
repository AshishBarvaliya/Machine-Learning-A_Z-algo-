#simple regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

# import linear regr
from sklearn.linear_model import LinearRegression
lrsor = LinearRegression()
lrsor.fit(X_train, y_train)

#predict regression
y_pred = lrsor.predict(X_test)

# plot the predict model for training set
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,lrsor.predict(X_train),color='blue')
plt.xlabel('ex')
plt.ylabel('slr')
plt.show()

# plot the predict model for testing set
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,lrsor.predict(X_train),color='blue')
plt.xlabel('ex')
plt.ylabel('slr')
plt.show()





