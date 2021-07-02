import pandas as pd 
from sklearn import tree  
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def main():
    data = pd.read_csv("tp3/glass.csv")

    del(data['Id'])

    x_train, x_test = train_test_split(data, test_size=0.33)

    y_train = x_train['Type']
    y_test = x_test['Type']

    del (x_train['Type'])
    del (x_test['Type'])

    classifier = tree.DecisionTreeClassifier(min_impurity_decrease= 0.02, max_depth=5, criterion='gini')
    classifier.fit(x_train, y_train)

    tree.export_graphviz(classifier, out_file='tree.dot', feature_names=['refractive index', 'Sodium', 'Magnesium', 'Aluminium', 'Silicon','Potassium','Calcium', 'Barium', 'Iron'])

    scoreTrain = classifier.score(x_train, y_train)
    scoreTest = classifier.score(x_test, y_test)

    print("Train : " + str(scoreTrain))
    print("Test : " + str(scoreTest))

    pred=classifier.predict(x_test)

    print(confusion_matrix(y_test, pred))

if __name__ == '__main__':
    main()
