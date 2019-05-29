
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#-- logostic regg
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

#confusion matrix -- used to check how many prediction data is true
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#10 -font- cross - validation
from sklearn.model_selection import cross_val_score
accuracie = cross_val_score( estimator=classifier, X=X_train, y=y_train, cv=10)
accuracie.mean()

# grid search
from sklearn.model_selection import GridSearchCV
parameter = [{'C':[1,10,100,1000], 'kernel':['linear']},
              {'C':[1,10,100,0.5], 'kernel':['rbf'], 'gamma':[0.5,0.7,0.9,0.99]}]
gridse = GridSearchCV(estimator=classifier,param_grid= parameter,scoring='accuracy', cv=10 )
gridse.fit(X_train,y_train)
best_acc = gridse.best_score_
best_params = gridse.best_params_

