# -*- coding: utf-8 -*-
"""
Plotting utilities for the trajectories of stochastic processes

@author: <alberto.suarez@uam.es>

"""

# Load packages
import numpy as np
from sklearn import datasets
import pickle
import pandas as pd


# ------------------- DATASETS -------------------


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


# ------------------- PICKLES -------------------


def pickle_dump(object, n_class, pickles_path="./pickles"):
    """
    Given an object, dumps it into a file.
    Uses the n_class to not overwrite different results.
    """
    file_path = "{}/clfs_{}_classes.p".format(pickles_path, n_class)
    with open(file_path, "w+b") as file:
        pickle.dump(object, file)


def pickle_load(n_class, pickles_path="./pickles"):
    """
    Loads a pickle from the respective file.
    """
    file_path = "{}/clfs_{}_classes.p".format(pickles_path, n_class)
    with open(file_path, "rb") as file:
        clf_results = pickle.load(file)
    return clf_results


# ------------------- TABLE COMPUTATION -------------------


def compute_error_tables(agglomerated_results_array, model_names):
    train_errors = np.array(
        [
            [1.0 - model["train_score"] for model in models]
            for models in agglomerated_results_array
        ]
    )
    train_error_table = pd.DataFrame(columns=model_names)
    train_error_table["Mean search time (per CV fold)"] = np.sum(train_errors, axis=0)
    train_error_table["Std search time (per CV fold)"] = np.std(train_errors, axis=0)

    test_errors = np.array(
        [
            [1.0 - model["test_score"] for model in models]
            for models in agglomerated_results_array
        ]
    )
    test_error_table = pd.DataFrame(columns=model_names)
    test_error_table["Mean search time (per CV fold)"] = np.sum(test_errors, axis=0)
    test_error_table["Std search time (per CV fold)"] = np.std(test_errors, axis=0)

    return (
        train_error_table,
        test_error_table,
    )


def compute_cv_error(clf_results_array, model_n, split=0):
    colum_names = ["params set {}".format(i) for i in len(clf_results_array[0])]
    results = clf_results_array[split][model_n].cv_results_

    cv_error_table = pd.DataFrame(columns=colum_names)
    cv_error_table["CV mean train error"] = 1.0 - results["mean_train_score"]
    cv_error_table["CV std train error"] = results["std_train_score"]
    cv_error_table["CV mean test error"] = 1.0 - results["mean_test_score"]
    cv_error_table["CV std test error"] = results["std_test_score"]
    return cv_error_table


def compute_time_tables(agglomerated_results_array, model_names):
    search_times = (
        np.array(
            [
                [model["search"] for model in models]
                for models in agglomerated_results_array
            ]
        )
        / 5.0
    )
    search_table = pd.DataFrame(columns=model_names)
    search_table["Mean search time (per CV fold)"] = np.sum(search_times, axis=0)
    search_table["Std search time (per CV fold)"] = np.std(search_times, axis=0)

    train_times = np.array(
        [
            [model["training"] for model in models]
            for models in agglomerated_results_array
        ]
    )
    training_table = pd.DataFrame(columns=model_names)
    training_table["Mean training time"] = np.sum(train_times, axis=0)
    training_table["Std training time"] = np.std(train_times, axis=0)

    prediction_times = np.array(
        [
            [model["prediction"] for model in models]
            for models in agglomerated_results_array
        ]
    )
    prediction_table = pd.DataFrame(columns=model_names)
    prediction_table["Mean training time"] = np.sum(prediction_times, axis=0)
    prediction_table["Std training time"] = np.std(prediction_times, axis=0)

    search_table, training_table, prediction_table
