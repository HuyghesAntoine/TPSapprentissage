import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree  
from sklearn.metrics import confusion_matrix

data = pd.read_csv('exam.csv')

def analyzeData():
    # Pour afficher le nombre d'exemples et le nombre de caractéristiques:  270 exemples et 13 caractéristiques 
    print(data.info())
    # Pour afficher les différentes statistiques de la base : 
    print(data.describe())
    # Pour affiche le nombre d'exemples de chaque classes : 150 de la classe 1 et 120 de la classe 2 
    print(data["N"].value_counts())
    df = pd.DataFrame(data)
    corrMatrix = df.corr()
    sn.heatmap(corrMatrix, annot=True)
    #plt.show()

def MLPclassif():
    xtrain, xtest = train_test_split(data, test_size = 0.33, shuffle=True)
    ytrain = xtrain['N']
    ytest = xtest['N']
    del (xtrain['N'])
    del (xtest['N'])

    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)

    classifier = MLPClassifier(max_iter=300)
    classifier.fit(xtrain, ytrain)

    scoreTrain = classifier.score(xtrain, ytrain)
    scoreTest = classifier.score(xtest, ytest)
    pred = classifier.predict(xtest)

    print("Learning with MLPClassifier :")
    print("Train : " + str(scoreTrain))
    print("Test : " + str(scoreTest))
    print(confusion_matrix(ytest, pred))

def knn():
    xtrain, xtest = train_test_split(data, test_size = 0.33, shuffle=True)
    ytrain = xtrain['N']
    ytest = xtest['N']
    del (xtrain['N'])
    del (xtest['N'])

    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)

    knn = KNeighborsClassifier(40)
    knn.fit(xtrain, ytrain)

    scoreTrain = knn.score(xtrain, ytrain)
    scoreTest = knn.score(xtest, ytest)
    pred=knn.predict(xtest)

    print("\nLearning with KNN :")
    print("Train : " + str(scoreTrain))
    print("Test : " + str(scoreTest))
    print(confusion_matrix(ytest, pred))

def treeClassif():
    xtrain, xtest = train_test_split(data, test_size=0.33)
    ytrain = xtrain['N']
    ytest = xtest['N']
    del (xtrain['N'])
    del (xtest['N'])

    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)

    classifier = tree.DecisionTreeClassifier(min_impurity_decrease= 0.02, max_depth=4)
    classifier.fit(xtrain, ytrain)
    tree.export_graphviz(classifier, out_file='tree.dot', feature_names=['A', 'B', 'C', 'D', 'E','F','G', 'H', 'I', 'J', 'K', 'L', 'M'])

    scoreTrain = classifier.score(xtrain, ytrain)
    scoreTest = classifier.score(xtest, ytest)
    pred=classifier.predict(xtest)

    print("\nLearning with TreeClassifier :")
    print("Train : " + str(scoreTrain))
    print("Test : " + str(scoreTest))
    print(confusion_matrix(ytest, pred))

analyzeData()
MLPclassif()
knn()
treeClassif()