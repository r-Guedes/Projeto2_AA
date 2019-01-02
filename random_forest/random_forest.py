from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, mean_squared_error
from sklearn import metrics
import numpy as np


import sys
sys.path.append("..") # Adds higher directory to python modules path.

from processing_dataset import get_dataset

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

if __name__ == '__main__':

    X, y = get_dataset()

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    clf = RandomForestClassifier(n_estimators=25, max_depth=3, max_features=8)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    y_pred_train = clf.predict(x_train)
    print(classification_report(y_test, y_pred))
    print("Accuracy Test:", metrics.accuracy_score(y_test, y_pred))
    print("Accuracy Train:", metrics.accuracy_score(y_train, y_pred_train))

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

    # use a full grid over all parameters
    param_grid = {"max_depth": [3, None],
                  "max_features": [1, 3, 8],
                  "min_samples_split": [2, 3, 10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)
    grid_search.fit(x_train, y_train)
    report(grid_search.cv_results_)

    for i in range(n):
        classifier = RandomForestClassifier(n_estimators=i + 1)
        classifier = classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_test)
        accuracy[i] = metrics.accuracy_score(y_test, predictions)

    plt.plot(range(1, n + 1), accuracy)
    plt.xlabel("Number of trees")
    plt.ylabel("Accuracy of prediction")
    plt.title("Effect of the number of trees on the prediction accuracy")
    plt.show()
    """