# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.linear_model import LinearRegression
lrsor = LinearRegression()
lrsor.fit(X, y)

#poly
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
lr_reg_2 = LinearRegression()
lr_reg_2.fit(x_poly,y)

plt.scatter(X,y,color='red')
plt.plot(X,lrsor.predict(X),color='blue')
plt.xlabel('no')
plt.ylabel('sals')
plt.show()

# for making more accurate 
x_grid =np.arange(min(X),max(X),0.1)
x_grid = x_grid.reshape(len(x_grid),1)
# for poly
plt.scatter(X,y,color='red')
plt.plot(X, lr_reg_2.predict(poly_reg.fit_transform(X)), color='blue')
plt.xlabel('no')
plt.ylabel('sals')
plt.show()
