import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV



import sys
sys.path.append("..") # Adds higher directory to python modules path.

from processing_dataset import get_dataset

if __name__ == '__main__':

    X, y = get_dataset()
    # Divide Dataset: 20% Test and 80% Train
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    clf = KNeighborsClassifier(n_neighbors=10, weights='uniform')
    clf = clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    y_pred_train = clf.predict(x_train)

    #A = clf.kneighbors_graph(X)

    print(classification_report(y_test, y_pred))
    print("Accuracy Test:", metrics.accuracy_score(y_test, y_pred))
    print("Accuracy Train:", metrics.accuracy_score(y_train, y_pred_train))

    """
    Finding the best K
    """

    k_range = list(range(1, 31))
    weight_options = ['uniform', 'distance']

    param_grid = dict(n_neighbors=k_range, weights=weight_options)

    grid = GridSearchCV(clf, param_grid, cv=10)
    grid.fit(X, y)
    print(grid.best_params_)

    grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
    plt.plot(k_range, grid_mean_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()






