import pandas as pd 
from sklearn import tree  
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    data = pd.read_csv("tp3/winequality-red.csv")

    x_train, x_test = train_test_split(data, test_size=0.33)

    y_train = x_train['quality']
    y_test = x_test['quality']

    del (x_train['quality'])
    del (x_test['quality'])

    regressor = tree.DecisionTreeRegressor(min_impurity_decrease= 0.02, max_depth=2, criterion='mse')
    regressor.fit(x_train, y_train)

    tree.export_graphviz(regressor, out_file='treeWine.dot', feature_names=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides','free sulfur dioxide','total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'])

    scoreTrain = regressor.score(x_train, y_train)
    scoreTest = regressor.score(x_test, y_test)
    pred=regressor.predict(x_test)

    print("Train : " + str(scoreTrain))
    print("Test : " + str(scoreTest))
    print("MAE : " + str(mean_absolute_error(y_test, pred)))
    print("MSE : " + str(mean_squared_error(y_test, pred)))

if __name__ == '__main__':
    main()
