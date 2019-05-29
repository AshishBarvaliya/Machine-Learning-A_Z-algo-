#import libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

#hieraicy in dendrogram 
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.show()

#fitting algo
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(X)

#visualization
plt.scatter(X[y_hc==0,0], X[y_hc==0,1],s=50, c='red', label='c1')
plt.scatter(X[y_hc==1,0], X[y_hc==1,1],s=50, c='blue', label='c2')
plt.scatter(X[y_hc==2,0], X[y_hc==2,1],s=50, c='green', label='c3')
plt.scatter(X[y_hc==3,0], X[y_hc==3,1],s=50, c='cyan', label='c4')
plt.scatter(X[y_hc==4,0], X[y_hc==4,1],s=50, c='pink', label='c5')
plt.legend()
plt.show()