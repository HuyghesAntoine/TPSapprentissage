from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns 

def label_encode(data, col):
        # Transforme un type catégorie en entier
        le = LabelEncoder()
        # On récupère tous les noms de catégories possibles
        unique_values = list(data[col].unique())
        le_fitted = le.fit(unique_values)
        # On liste l'ensemble des valeurs
        values = list(data[col].values)
        # On transforme les catégories en entier
        values_transformed = le.transform(values)
        # On fait le remplacement de la colonne dans le dataframe d'origine
        data[col] = values_transformed

def classif():
    data = pd.read_csv("tp5/human_resources.csv")
    label_encode(data,'sales')
    label_encode(data,'salary')

    xtrain, xtest = train_test_split(data, test_size = 0.33, shuffle=True)
        
    ytrain = xtrain['left']
    ytest = xtest['left']

    del (xtrain['left'])
    del (xtest['left'])

    classifier = MLPClassifier()
    classifier.fit(xtrain, ytrain)

    scoreTrain = classifier.score(xtrain, ytrain)
    scoreTest = classifier.score(xtest, ytest)
    pred = classifier.predict(xtest)

    print("Train : " + str(scoreTrain))
    print("Test : " + str(scoreTest))
    print(confusion_matrix(ytest, pred))

    analyze_good_employees(data)

def regress():
    data = pd.read_csv("tp5/hour.csv")
    del data['instant']
    del data['dteday']

    xtrain, xtest = train_test_split(data, test_size = 0.33, shuffle=True)
        
    ytrain = xtrain['cnt']
    ytest = xtest['cnt']

    del (xtrain['cnt'])
    del (xtest['cnt'])

    regress = MLPRegressor()
    regress.fit(xtrain, ytrain)
    scoreTrain = regress.score(xtrain, ytrain)
    scoreTest = regress.score(xtest, ytest)
    pred = regress.predict(xtest)
    
    print("Train : " + str(scoreTrain))
    print("Test : " + str(scoreTest))
    print("MAE : " + str(mean_absolute_error(ytest, pred)))
    print("MSE : " + str(mean_squared_error(ytest, pred)))

def analyze_good_employees(data):
    averages = data.mean()
    average_last_evaluation = averages['last_evaluation']
    average_project = averages['number_project']
    average_montly_hours = averages['average_montly_hours']
    average_time_spend = averages['time_spend_company']

    good_employees = data[data['last_evaluation'] > average_last_evaluation]
    good_employees = good_employees[good_employees['number_project'] > average_project]
    good_employees = good_employees[good_employees['average_montly_hours'] > average_montly_hours]
    good_employees = good_employees[good_employees['time_spend_company'] > average_time_spend]

    sns.set()
    plt.figure(figsize=(15, 8))
    plt.hist(data['left'])
    print(good_employees.shape)
    sns.heatmap(good_employees.corr(), vmax=0.5, cmap="PiYG")
    plt.title('Correlation matrix')
    plt.show()


regress()

