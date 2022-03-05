# -*- coding: utf-8 -*-
"""
Plotting utilities for the trajectories of stochastic processes

@author: <alberto.suarez@uam.es>

"""

# Load packages
import numpy as np
from sklearn import datasets
import pickle


def create_S_dataset(n_instances=1000, shuffle=True):
    """Generates a dataset with a S shape"""
    ## Generate data
    # 3-D data
    X, y = datasets.make_s_curve(n_instances, noise=0.1)

    # This only shuffles the X values respect to the labels (y).
    # Since the labels are not used, this sorting
    if shuffle:
        X = X[np.argsort(y)]

    # Reshape if necessary
    if X.ndim == 1:
        X = X[:, np.newaxis]

    return X, y


def get_digits_dataset(n_class=10):
    # The digits dataset
    X, y = datasets.load_digits(n_class=n_class, return_X_y=True)

    # The features are initially in [0, 16.0]
    X = X / 16.0

    # Center data
    X -= X.mean(axis=0)

    return X, y


def pickle_dump(clf_results, n_class, pickles_path="./pickles"):
    """
    Given a clf_results object, dumps it into a file.
    Uses the n_class to not overwrite different results.
    """
    file_path = "{}/clfs_{}_classes.p".format(pickles_path, n_class)
    with open(file_path, "w+b") as file:
        pickle.dump(clf_results, file)


def pickle_load(n_class, pickles_path="./pickles"):
    """
    Loads a pickle from the respective file.
    """
    file_path = "{}/clfs_{}_classes.p".format(pickles_path, n_class)
    with open(file_path, "rb") as file:
        clf_results = pickle.load(file)
    return clf_results
