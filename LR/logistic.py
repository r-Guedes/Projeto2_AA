from sklearn.model_selection import train_test_split, cross_val_score
from tunning_hyperparameters import logistic_tunning
from train_model import logistic_train
from test_model import test
from evaluate_model import evaluate
from sklearn.linear_model import LogisticRegression

import sys
sys.path.append("..") # Adds higher directory to python modules path.
from processing_dataset import get_dataset

if __name__ == '__main__':

    X, y = get_dataset()

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    best_parameters = logistic_tunning(x_train, y_train)

    trained_model_regularized = logistic_train(x_train, y_train, penalty=best_parameters["model_regularized"]["Penalty"], C=best_parameters["model_regularized"]["C"], solver=best_parameters["model_regularized"]["Solver"], multi_class=best_parameters["model_regularized"]["MultiClass"], max_iter=1000)
    trained_model_non_regularized = logistic_train(x_train, y_train, penalty=best_parameters["model_non_regularized"]["Penalty"], C=best_parameters["model_non_regularized"]["C"], solver=best_parameters["model_non_regularized"]["Solver"], multi_class=best_parameters["model_non_regularized"]["MultiClass"], max_iter=1000)

    lg = LogisticRegression(penalty=best_parameters["model_regularized"]["Penalty"], C=best_parameters["model_regularized"]["C"], solver=best_parameters["model_regularized"]["Solver"], multi_class=best_parameters["model_regularized"]["MultiClass"], max_iter=1000)
    cvs = cross_val_score(lg,x_train,y_train,cv=4)

    # Evaluate the model
    print("\n Evaluate the model \n")
    print("\nRegularized:")
    evaluate(trained_model_regularized, x_train, y_train)
    print("\n Cross_validation")
    print(cvs)
    print("\nNon Regularized:")
    evaluate(trained_model_non_regularized, x_train, y_train)

    # Test the model
    print("\n Test the model \n")
    print("\nRegularized:")
    test(trained_model_regularized, x_test, y_test)
    print("\nNon Regularized:")
    test(trained_model_non_regularized, x_test, y_test)
