import numpy
from sklearn.model_selection import train_test_split
from optimize import optimize
from train_wine_model import nn_train
from evaluate_wine_model import evaluate
from test_model import test

from processing_dataset import get_dataset



if __name__ == '__main__':

    X, y = get_dataset()

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
	
    x_train_60, x_val, y_train_60, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=5)
    # best_parameters
    best_parameters = optimize(x_train_60, y_train_60, x_val, y_val)

    # best_parameters
    #best_parameters = optimize(x_train, y_train, x_test, y_test)


    # Pass the best parameters to train, and the Train Data
    trained_model = nn_train(x_train, y_train, x_test, y_test, best_parameters)
	
    

    # Evaluate the model
    evaluate(trained_model, x_train, y_train)

    # Test the model
    test(trained_model, x_test, y_test)