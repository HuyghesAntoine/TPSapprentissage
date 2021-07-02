import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

data = pd.read_csv('iris.csv')

"""
# Combien de classe : 3
print(data["Species"].value_counts())
# Combien de cracatèristique descriptives : 4
print(data.describe())
# Combien d'exemples : 150
print(data.tail())
# Combien d'exemples de chaque classe : 50 de chaque
print(data["Species"].value_counts())
# Comment sont organisés les exemples : organisé par espèces
"""

train, test = train_test_split(data, test_size=0.33, shuffle=True)

xtrain = train
ytrain = train["Species"]
del(xtrain['Species'])
del(xtrain['Id'])

xtest = test
ytest = test["Species"]
del(xtest['Species'])
del(xtest['Id'])

knn = KNeighborsClassifier(5)
knn.fit(xtrain, ytrain)

scoreTrain = knn.score(xtrain, ytrain)
scoreTest = knn.score(xtest, ytest)

print("Train : " + str(scoreTrain))
print("Test : " + str(scoreTest))

pred=knn.predict(xtest)

print(confusion_matrix(ytest, pred))

