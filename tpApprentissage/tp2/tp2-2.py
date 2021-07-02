import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('auto-mpg2.csv')

'''
print(data.shape)
print(data.info())
print(data.describe())
'''

train, test = train_test_split(data, test_size=0.5, shuffle=True)

xtrain = train
ytrain = train["mpg"]
del(xtrain['name'])
del(xtrain['mpg'])

xtest = test
ytest = test["mpg"]
del(xtest['name'])
del(xtest['mpg'])

scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

knn = KNeighborsRegressor(5)
knn.fit(xtrain,ytrain)

scoreTrain = knn.score(xtrain, ytrain)
scoreTest = knn.score(xtest, ytest)

print("Train : " + str(scoreTrain))
print("Test : " + str(scoreTest))

pred=knn.predict(xtest)

print("MAE : " + str(mean_absolute_error(ytest, pred)))
print("MSE : " + str(mean_squared_error(ytest, pred)))

