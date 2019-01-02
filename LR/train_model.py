from sklearn.linear_model import LogisticRegression


# # ___________________ Logistic Regression HyperParameters Tunning __________________ #
def logistic_train(X_train, y_train, penalty, C, solver, multi_class, max_iter):

    lg = LogisticRegression(penalty=penalty, C=C, solver=solver, multi_class=multi_class, max_iter=max_iter)
    lg.fit(X_train, y_train)
    return lg