#-----------data processing--------- 

#import libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#read dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

#handle missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(x[:,1:3])
x[:,1:3]= imputer.transform(x[:,1:3])

#encoding categatorical data
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
labelencoder=LabelEncoder()
x[:,0] = labelencoder.fit_transform(x[:,0])
onehot = OneHotEncoder(categorical_features=[0])
x=onehot.fit_transform(x).toarray()
labelencoder=LabelEncoder()
y = labelencoder.fit_transform(y)

#split into train and test sets
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#feature scaling 
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)





















