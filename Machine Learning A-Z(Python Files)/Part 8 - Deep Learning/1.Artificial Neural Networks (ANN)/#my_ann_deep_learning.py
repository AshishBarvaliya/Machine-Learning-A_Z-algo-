# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder1 = LabelEncoder()
X[:, 2] = labelencoder1.fit_transform(X[:, 2])
labelencoder2 = LabelEncoder()
X[:, 1] = labelencoder2.fit_transform(X[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#creating ann
from keras.models import Sequential
from keras.layers import Dense

#initialize ANN 
classifier = Sequential()

#create hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu',input_dim=11))
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))

#output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

#compile
classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

#fit the model
classifier.fit(X_train,y_train,batch_size = 10, epochs=100)

#predict
y_pred=classifier.predict(X_test)
y_pred = (y_pred>0.5)
#confusion matrix -- used to check how many prediction data is true
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
