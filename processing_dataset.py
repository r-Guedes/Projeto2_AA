import visuals as vs
import pandas as pd
import seaborn as sns
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np


def get_dataset():
    dataset = pd.read_csv("../winequality-white.csv", sep=";")

    outliers = []

    for feature in dataset.keys():
        Q1 = np.percentile(dataset[feature], q=25)
        Q3 = np.percentile(dataset[feature], q=75)
        interquartile_range = Q3 - Q1
        step = 1.5 * interquartile_range
        #print("Data points considered outliers for the feature '{}':".format(feature))
        outliers_obj = dataset[~((dataset[feature] >= Q1 - step) & (dataset[feature] <= Q3 + step))]
        #display(outliers_obj)
        outliers = list(set(outliers + list(outliers_obj.index)))

    dataset = dataset.drop(dataset.index[outliers]).reset_index(drop=True)

    # Parse Data into 3 Categories
    y = dataset.quality
    new_y = []

    for each in y:
        if 0 <= each <= 4:
            new_y.append(0)
        elif each == 6:
            new_y.append(1)
        else:
            new_y.append(2)

    y = new_y
    X = dataset.drop('quality', axis=1)
    X = X.drop('total sulfur dioxide', axis=1)
    X = X.drop('fixed acidity', axis=1)
    X = X.drop('citric acid', axis=1)

    #visualize_dataset(dataset)
    return X, y

def visualize_dataset(dataset):
    vs.distribution(dataset, "quality")

    correlation = dataset.corr()
    display(correlation)
    plt.figure(figsize=(14, 12))
    heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")
    plt.show()
