import numpy
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

if __name__ == '__main__':

    # Load Dataset
    dataset = numpy.loadtxt("../winequality-white.csv", delimiter=";", skiprows=1)

    # Parse Data into 3 Categories
    y = dataset[:, 11]
    new_y = []

    for each in y:
        if 0 <= each <= 4:
            # 0, 1, 2, 3, 4
            new_y.append(0)
        #elif 5 <= each <= 6:
        elif each == 6:
            # 5, 6
            new_y.append(1)
        else:
            # 7, 8, 9, 10
            new_y.append(2)

    y = new_y
    X = dataset[:, 0:11]

    # Divide Dataset: 20% Test and 80% Train
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
	
    clf = tree.DecisionTreeClassifier(criterion='gini')
    clf = clf.fit(x_train,y_train)
    
    train_pred = clf.predict(x_train)
    test_pred = clf.predict(x_test)
    
    
    print("Accuracy train set: ", accuracy_score(y_train, train_pred))
    print("Accuracy test set: ", accuracy_score(y_test, test_pred))
    
    
    
    

