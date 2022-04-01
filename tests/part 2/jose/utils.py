import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


my_cmap = plt.cm.Spectral


def scatter_2D(X, y, title="", equal_axis=False):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=my_cmap)
    if equal_axis:
        ax.axis("equal")
    ax.set_title(title)
    plt.show()


def scatter_3D(X, y, title="", figsize=(12, 8)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=my_cmap)
    ax.set_title(title)
    ax.view_init(15, -72)
    plt.tight_layout()
    plt.show()


def plot_decision_boundary(X, y, kmeans_model, title=""):
    # Reference https://datascience.stackexchange.com/questions/ \
    # 53918/is-there-a-way-to-put-a-separate-line-between-clusters \
    # -for-k-means-clustering

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = 0.002  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans_model.predict(np.c_[xx.ravel(), yy.ravel()])
    color = np.copy(Z)
    color = np.where(Z == 1, "skyblue", color)
    color = np.where(Z == 0, "tomato", color)

    # Put the result into a color plot
    plt.figure(1)
    plt.clf()
    plt.scatter(
        xx,
        yy,
        c=color,
        # interpolation="nearest",
        # extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        # aspect="auto",
        # origin="lower",
        alpha=0.025,
    )

    # Plot the points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=my_cmap)

    if title != "":
        plt.title(title)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()
