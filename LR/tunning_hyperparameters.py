from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np


# # ___________________ Logistic Regression HyperParameters Tunning __________________ #
def logistic_tunning(X_train, y_train):

    lg = LogisticRegression()

    """
        Non-Regularized Cost Function
    """

    penalty = ['l1']  # Non-regularized cost function
    C = np.logspace(0, 4, 10)  # Regularization Strength - smaller values means stronger regularization
    solver = ['liblinear', 'saga']
    multi_class = ['ovr']

    hyperparameters = dict( penalty=penalty, solver=solver, multi_class=multi_class)

    clf = GridSearchCV(lg, hyperparameters, cv=4, verbose=0)

    clf.fit(X_train, y_train)

    print('Best Penalty:', clf.best_estimator_.get_params()['penalty'])
    print('Best C:', clf.best_estimator_.get_params()['C'])
    print('Multi Class:', clf.best_estimator_.get_params()['multi_class'])
    print('Solver:', clf.best_estimator_.get_params()['solver'])

    non_regularized = {"model_non_regularized": {"Penalty": clf.best_estimator_.get_params()['penalty'], "C": clf.best_estimator_.get_params()['C'], "MultiClass": clf.best_estimator_.get_params()['multi_class'], "Solver": clf.best_estimator_.get_params()['solver']}}

    """
        Regularized Cost Function
    """

    penalty = ['l2']  # Regularized cost function
    C = np.logspace(0, 4, 10)  # Regularization Strength - smaller values means stronger regularization
    solver = ['newton-cg', 'lbfgs', 'sag']
    multi_class = ['ovr', 'multinomial']

    hyperparameters = dict(C=C, penalty=penalty, solver=solver, multi_class=multi_class)

    clf = GridSearchCV(lg, hyperparameters, cv=5, verbose=0)

    clf.fit(X_train, y_train)

    print('Best Penalty:', clf.best_estimator_.get_params()['penalty'])
    print('Best C:', clf.best_estimator_.get_params()['C'])
    print('Multi Class:', clf.best_estimator_.get_params()['multi_class'])
    print('Solver:', clf.best_estimator_.get_params()['solver'])

    regularized = {"model_regularized": {"Penalty": clf.best_estimator_.get_params()['penalty'], "C":clf.best_estimator_.get_params()['C'], "MultiClass": clf.best_estimator_.get_params()['multi_class'], "Solver": clf.best_estimator_.get_params()['solver']}}

    result = dict(list(non_regularized.items()) + list(regularized.items()))

    return result