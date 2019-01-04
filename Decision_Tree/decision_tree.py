from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

sys.path.append("..") # Adds higher directory to python modules path.

from processing_dataset import get_dataset

if __name__ == '__main__':

    X, y = get_dataset()

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_split=10)

    clf = clf.fit(x_train,y_train)
    
    train_pred = clf.predict(x_train)
    test_pred = clf.predict(x_test)
    
    print(classification_report(y_test, test_pred))
    print("Accuracy train set: ", accuracy_score(y_train, train_pred))
    print("Accuracy test set: ", accuracy_score(y_test, test_pred))
    print("Mean squared error - test set:", mean_squared_error(y_test, test_pred))
    print("Mean squared error - training set:", mean_squared_error(y_train, train_pred))
    print("Cross_validation_evaluate: ", cross_val_score(clf, x_train, y_train, cv=10).mean())

    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    Image(graph.write_pdf("DT.pdf"))

    # Create CV training and test scores for various training set sizes
    train_sizes, train_scores, test_scores = learning_curve(clf,
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

    parameters = {'min_samples_split': range(10, 500, 20), 'max_depth': range(1, 20), 'criterion': ['gini', 'entropy']}
    grid = GridSearchCV(tree.DecisionTreeClassifier(), parameters, n_jobs=4)
    grid = grid.fit(X, y)
    print(grid.best_params_)

    grid_mean_scores = [{'max_depth': result[0]['max_depth'],'cv': result.mean_validation_score} for result in grid.grid_scores_]

    # Plot Max D
    #
    #  plt.plot(range(10, 500, 20), grid_mean_scores)
    plt.xlabel('Maximum Depth')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()


    
    
    

