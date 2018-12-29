import numpy
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
import sys
sys.path.append("..") # Adds higher directory to python modules path.

from processing_dataset import get_dataset

if __name__ == '__main__':

    X, y = get_dataset()

    # Divide Dataset: 20% Test and 80% Train
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
	
    clf = tree.DecisionTreeClassifier(criterion='gini')
    clf = clf.fit(x_train,y_train)
    
    train_pred = clf.predict(x_train)
    test_pred = clf.predict(x_test)
    
    
    print("Accuracy train set: ", accuracy_score(y_train, train_pred))
    print("Accuracy test set: ", accuracy_score(y_test, test_pred))
    
    
    
    

