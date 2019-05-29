#import libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

#k-means
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.show()

kmeans = KMeans(n_clusters=5, random_state=0)
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans==0,0], X[y_kmeans==0,1],s=50, c='red', label='c1')
plt.scatter(X[y_kmeans==1,0], X[y_kmeans==1,1],s=50, c='blue', label='c2')
plt.scatter(X[y_kmeans==2,0], X[y_kmeans==2,1],s=50, c='green', label='c3')
plt.scatter(X[y_kmeans==3,0], X[y_kmeans==3,1],s=50, c='cyan', label='c4')
plt.scatter(X[y_kmeans==4,0], X[y_kmeans==4,1],s=50, c='pink', label='c5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow')
plt.legend()
plt.show()