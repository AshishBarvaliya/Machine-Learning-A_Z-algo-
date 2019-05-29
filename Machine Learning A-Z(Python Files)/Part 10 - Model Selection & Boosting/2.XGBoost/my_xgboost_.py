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

#xgboosting 
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train,y_train)

#predict
y_pred=classifier.predict(X_test)

#confusion matrix -- used to check how many prediction data is true
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#10 -font- cross - validation
from sklearn.model_selection import cross_val_score
accuracie = cross_val_score( estimator=classifier, X=X_train, y=y_train, cv=10)
accu = accuracie.mean()
