from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np

import sys
sys.path.append("..") # Adds higher directory to python modules path.

from processing_dataset import get_dataset

if __name__ == '__main__':

    X, y = get_dataset()
    
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    
    # Divide Dataset: 20% Test and 80% Train
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


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
    print("Cross_validation_evaluate: ", cross_val_score(gnb, x_train, y_train, cv=10))
    print("Cross_validation_evaluate mean: ", cross_val_score(gnb, x_train, y_train, cv=10).mean())
    
    # Create CV training and test scores for various training set sizes
    train_sizes, train_scores, test_scores = learning_curve(gnb,
                                                            X,
                                                            y,
                                                            cv=10,
                                                            n_jobs=-1,
                                                            train_sizes=np.linspace(0.01, 1.0, 50))

    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Draw lines
    plt.plot(train_sizes, train_mean, '--', color="#111111", label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

    # Draw bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

    # Create plot
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()