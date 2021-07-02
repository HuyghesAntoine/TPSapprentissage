import matplotlib.pyplot as plt 
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

np.random.seed(1337)
nb = 15
nb2 = 50

x =  np.sort( np.random.uniform(-3, 10, nb) )
y = 10.0*np.sin(x)/x + np.random.normal(0, 1, nb)
x = x.reshape(-1,1)
plt.plot(x,y,'.')

x_test = np.sort(np.random.uniform(-3,10,nb2))
y_test = 10.0*np.sin(x_test)/x_test + np.random.normal(0,1,nb2)
x_test = x_test.reshape(-1,1)

x_plot = np.linspace(-3,10,nb)
x_plot = x_plot.reshape(-1,1)

degre = np.array([1,3,6,9,12])
polynome= []
for i in range(0,len(degre)):

    poly = make_pipeline(PolynomialFeatures(degre[i]), Ridge())
    poly.fit(x,y)
    y_plot = poly.predict(x_plot)
    plt.plot(x_plot, y_plot,label=degre[i])
    polynome.append(poly)
    r = poly.score(x_test, y_test)

    predTrain = poly.predict(x)
    predTest = poly.predict(x_test)

    residualErr = np.square(y-predTrain)
    residualErrTest = np.square(y_test-predTest)

    print ("Pour le deg " + str(degre[i]) + " : ")
    print ("RSS train : " + str(np.sum(residualErr)))
    print ("RSS test : " + str(np.sum(residualErrTest)))
    print ("R squared : " + str(r))

plt.legend()
plt.ylim(-5, 15)
plt.show()

#Le dégré 12 a bcp trop surappris
# 1 a sous appris
# 3 = le meilleur qu'on ait

# Classification :  valeurs discretes
# regression : valeurs continues