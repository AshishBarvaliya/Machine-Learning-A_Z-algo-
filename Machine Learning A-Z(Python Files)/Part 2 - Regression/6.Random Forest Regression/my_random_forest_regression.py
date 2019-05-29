import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=300,random_state=0)
reg.fit(X,y)

y_pred = reg.predict(6.5)
# for making more accurate 
x_grid =np.arange(min(X),max(X),0.1)
x_grid = x_grid.reshape(len(x_grid),1)
# for poly
plt.scatter(X,y,color='red')
plt.plot(x_grid, reg.predict(x_grid), color='blue')
plt.xlabel('no')
plt.ylabel('sals')
plt.show()
