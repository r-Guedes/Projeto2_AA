from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import numpy as np

import sys
sys.path.append("..") # Adds higher directory to python modules path.

from processing_dataset import get_dataset

if __name__ == '__main__':

    X, y = get_dataset()
    # Divide Dataset: 20% Test and 80% Train
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    print("--------------------------------- GAUSSIAN ---------------------------------")

    gnb = GaussianNB()
    gnb.fit(x_train, y_train)

    y_pred = gnb.predict(x_test)
    y_pred_train = gnb.predict(x_train)

    print(classification_report(y_test, y_pred))
    print("Accuracy Test:", metrics.accuracy_score(y_test, y_pred))
    print("Accuracy Train:", metrics.accuracy_score(y_train, y_pred_train))

    print("Mean squared error - test set:", mean_squared_error(y_test, y_pred))
    print("Mean squared error - training set:", mean_squared_error(y_train, y_pred_train))
    print("Cross_validation_evaluate: ", cross_val_score(gnb, x_train, y_train, cv=10).mean())