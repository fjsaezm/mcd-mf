import pickle
import warnings
from collections import defaultdict

import numpy as np
from sklearn.model_selection import GridSearchCV


def save_object(object, filename):
    with open(filename, 'wb') as f:
        pickle.dump(object, f)


def load_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def print_best_param_score(clf):
    for param in clf.best_params_.keys():
        param_name = param.split("__")[-1]
        print("\t - {} : {}".format(param_name, clf.best_params_[param]))
    print("Mean cross-validated score of the best_estimator")

    print("\t Train: " + str(clf.best_score_))
    print("---")


def grid_search(X_train, y_train, classifiers, param_search_space, iteration=None, verbose=False):

    iteration_results = defaultdict(str)
    for (name, model), param_space in zip(classifiers, param_search_space):

        clf = GridSearchCV(
            model,
            param_space,
            n_jobs=-1
        )

        # Find best classifier
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            clf.fit(X_train, y_train)

        if verbose:
            print(name)
            print()
            print("Best parameters set found on development set:")
            for param in clf.best_params_.keys():
                param_name = param.split("__")[-1]
                print("\t - {} : {}".format(param_name, clf.best_params_[param]))

            print("Scores of best parameters")
            print("\t Train: " + str(clf.best_score_))
            print("-------------------")

        iteration_results[name] = clf

    save_object(iteration_results, "results-optimal/iteration-{}.p".format(iteration))


def find_best_model(model_list):
    index = np.argmax(np.array([model.best_score_ for model in model_list]))

    return index
