
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv',header = None)

#apiori require 2d list so 
transactions = []
for n in range(0,7501):
    transactions.append([str(dataset.values[n,i]) for i in range(0,20)])

#apply algo
from apyori import apriori
rules = apriori(transactions,min_support=0.003, min_confidence=0.2, 
                min_lift=3, min_length=2)

#visualization
results = list(rules)
for i in range(0,6):
    print(results[i])
    print('**************')