#!/usr/bin/python3 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X= np.array([[6],[8],[10],[14],[18]])
Y= [7,9,13,17.5,18]
points_x = np.arange(0.0, 25.0, 0.1).reshape(-1,1)
plt.plot(X,Y,'.')

regr = LinearRegression()
regr.fit(X,Y)
pred = regr.predict(X)
plt.plot(X, pred)

residualErr = np.square(Y-pred)
print ("Residual sum : " + str(np.sum(residualErr)))

plt.show()