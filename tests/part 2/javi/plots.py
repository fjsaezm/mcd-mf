"""
Plot functions file used in the Functional Methods exam
Author: <franciscojavier.saez@estudiante.uam.es>
"""

# Needed libraries for this file
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score



my_cmap = plt.cm.Spectral


def scatter_2D(X, y, title="", equal_axis=False, ax=None, hide_ticks=False):
    """
    Plots a 2D scatter with labels. An axis can be passed
    """
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot()
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=my_cmap)
    if equal_axis:
        ax.axis('equal')
    ax.set_title(title)

    # Hide ticks if needed
    if hide_ticks:
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    # If single figure
    if ax == None:
        plt.show()


def scatter_3D(X, y, title=""):
    """
    Plots a 3D scatter with labels.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=my_cmap)
    ax.set_title(title)
    ax.view_init(26,51)
    plt.show()

def plot_classifier(X,y, clf,ax = None, title = ""):
    """
    Plots a 2D scatter of the data X and the prediction regions
    from classifier clf. 
    """
    
    # create a mesh to plot in
    h = .001  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 0.05, X[:, 0].max() + 0.05
    y_min, y_max = X[:, 1].min() - 0.05, X[:, 1].max() + 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
    
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=my_cmap, alpha=0.2)
    # Plot points
    scatter_2D(X,y,ax = ax, hide_ticks = True)
    
    acc = accuracy_score(clf.predict(X),y)
    ax.set_title(title + ". Acc = {}".format(acc))
    