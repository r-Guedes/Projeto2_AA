from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, mean_squared_error
from sklearn import metrics
import numpy as np


import sys
sys.path.append("..") # Adds higher directory to python modules path.

from processing_dataset import get_dataset

if __name__ == '__main__':

    X, y = get_dataset()

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    clf = RandomForestClassifier(n_estimators=25, max_features=6, max_depth=3, min_samples_split=3)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    y_pred_train = clf.predict(x_train)
    print(classification_report(y_test, y_pred))
    print("Accuracy Test:", metrics.accuracy_score(y_test, y_pred))
    print("Accuracy Train:", metrics.accuracy_score(y_train, y_pred_train))
    print("Cross_validation_evaluate: ", cross_val_score(clf, x_train, y_train, cv=10).mean())

    """
    param_dist = {"max_depth": [3, None],
                  "max_features": sp_randint(1, 8),
                  "min_samples_split": sp_randint(2, 11),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search, cv=5)
    random_search.fit(x_train, y_train)
    report(random_search.cv_results_)
    """
    # use a full grid over all parameters
    param_grid = {"max_depth": [1, 2, 3, None], "max_features": [4, 5, 6, 7, 8], "min_samples_split": [2, 3, 10], "bootstrap": [True, False], "criterion": ["gini", "entropy"], "n_estimators": range(1,30)}

    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=4)
    grid_search.fit(X, y)
    print(grid_search.best_params_)

    n = 30
    accuracy = [None]*n
    for i in range(n):
        classifier = RandomForestClassifier(max_features=6, max_depth=3, min_samples_split=3, n_estimators=i + 1)
        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_test)
        accuracy[i] = cross_val_score(classifier, x_train, y_train, cv=4)

    plt.plot(range(1, n + 1), accuracy)
    plt.xlabel("Number of trees")
    plt.ylabel("Accuracy of prediction")
    plt.title("Effect of the number of trees on the prediction accuracy")
    plt.show()

    print("teste")
