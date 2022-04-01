import matplotlib.pyplot as plt

my_cmap = plt.cm.Spectral


def scatter_2D(X, y, title="", equal_axis=False):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=my_cmap)
    if equal_axis:
        ax.axis("equal")
    ax.set_title(title)
    plt.show()


def scatter_3D(X, y, title=""):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=my_cmap)
    ax.set_title(title)
    plt.show()
