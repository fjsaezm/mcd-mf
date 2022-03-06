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

""" Computes single table """


def _create_single_table(agglomerated_results_array, field, index_names, model_names):
    field_values = np.array(
        [[model[field] for model in models] for models in agglomerated_results_array]
    )
    return pd.DataFrame(
        [np.mean(field_values, axis=0), np.std(field_values, axis=0)],
        columns=model_names,
        index=index_names,
    )


def compute_error_tables(agglomerated_results_array, model_names):
    train_error_table = _create_single_table(
        agglomerated_results_array,
        "train_score",
        index_names=["Mean train error", "Std train error"],
        model_names=model_names,
    )

    test_error_table = _create_single_table(
        agglomerated_results_array,
        "test_score",
        index_names=["Mean test error", "Std test error"],
        model_names=model_names,
    )

    train_error_table.loc["Mean train error"] = (
        1.0 - train_error_table.loc["Mean train error"]
    )
    test_error_table.loc["Mean test error"] = (
        1.0 - test_error_table.loc["Mean test error"]
    )

    return (
        train_error_table,
        test_error_table,
    )


def compute_cv_error(clf_results_array, model_names):
    mean_scores = np.array(
        [
            [np.mean(1.0 - clf.cv_results_["mean_test_score"]) for clf in clfs]
            for clfs in clf_results_array
        ]
    )
    std_of_means_scores = np.array(
        [
            [np.std(1.0 - clf.cv_results_["mean_test_score"]) for clf in clfs]
            for clfs in clf_results_array
        ]
    )

    mean_cv_scores_table = pd.DataFrame(mean_scores, columns=model_names)
    std_cv_scores_table = pd.DataFrame(std_of_means_scores, columns=model_names)
    return mean_cv_scores_table, std_cv_scores_table


def compute_time_tables(agglomerated_results_array, model_names):

    # Divide search table by 5 to compute per each of the 5-folds
    search_table = _create_single_table(
        agglomerated_results_array,
        "search",
        index_names=["Mean search time (per CV fold)", "Std search time (per CV fold)"],
        model_names=model_names,
    ).div(5.0)
    training_table = _create_single_table(
        agglomerated_results_array,
        "training",
        index_names=["Mean training time", "Std training time"],
        model_names=model_names,
    )
    prediction_table = _create_single_table(
        agglomerated_results_array,
        "prediction",
        index_names=["Mean prediction time", "Std prediction time"],
        model_names=model_names,
    )

    return search_table, training_table, prediction_table
