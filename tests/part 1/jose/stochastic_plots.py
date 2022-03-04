# -*- coding: utf-8 -*-
"""
Plotting utilities for the trajectories of stochastic processes

@author: <alberto.suarez@uam.es>

"""

# Load packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from typing import Callable, Union, Type
from kernel_approximation import RandomFeaturesSampler, NystroemFeaturesSampler


fig_size = (8, 6)


def plot_trajectories(
    t, X, max_trajectories=20, fig_num=1, fontsize=14, mean_color="k"
):
    """Plots a sample of trajectories and their mean"""

    M, _ = np.shape(X)

    # Plot trajectories
    M = np.min((M, max_trajectories))
    fig, ax = plt.subplots(1, 1, num=fig_num, figsize=fig_size)
    ax.plot(t, X[:M, :].T, linewidth=1)
    ax.set_xlabel("t", fontsize=fontsize)
    ax.set_ylabel("X(t)", fontsize=fontsize)
    ax.set_title("Simulation", fontsize=fontsize)

    # Plot mean
    ax.plot(t, np.mean(X, axis=0), linewidth=3, color=mean_color)

    return fig, ax


def plot_pdf(X, pdf, max_bins=50, ax=None, fig_size=fig_size, fontsize=14):
    """Compare pdf with the normalized histogram.
    The normalized histogram is an empirical estimate of the pdf.
    """

    # Plot histogram
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=fig_size)
    n_samples = len(X)
    n_bins = np.min((np.int(np.sqrt(n_samples)), max_bins))

    ax.hist(
        X, bins=n_bins, density=True, label="Density estimation", color="deepskyblue"
    )

    ax.set_xlabel("$x$", fontsize=fontsize)
    ax.set_ylabel("pdf($x$)", fontsize=fontsize)

    # Compare with exact distribution
    n_plot = 1000
    x_plot = np.linspace(np.min(X), np.max(X), n_plot)
    y_plot = pdf(x_plot)
    ax.plot(x_plot, y_plot, linewidth=2, color="tomato", label="Derived pdf")

    ax.legend()


def plot_pdf_3d(X, max_bins=50, lim=3.5, fig_size=fig_size, title=None):
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(projection="3d")
    hist, xedges, yedges = np.histogram2d(
        X[:, 0], X[:, 1], bins=max_bins, range=[[-lim, lim], [-lim, lim]]
    )

    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 0.5 * np.ones_like(zpos)
    dz = hist.ravel() / len(X)

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color="deepskyblue")

    if title is not None:
        ax.set_title(title)
    plt.show()


def plot_kernel_error(
    X: np.ndarray,
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    features_sampler_class: Union[
        Type[RandomFeaturesSampler], Type[NystroemFeaturesSampler]
    ],
    features_sampler_kwargs: np.ndarray,
    n_features: np.ndarray,
    kernel_name: str,
) -> None:
    # Initialize variables
    kernel_matrix = kernel(X, X)
    errors = np.array([], dtype=float)

    for n_f in n_features:
        # Create sampler
        features_sampler = features_sampler_class(
            **features_sampler_kwargs, n_features_sampled=n_f
        )

        # Compute approximation
        X_features = features_sampler.fit_transform(X)
        kernel_matrix_approx = X_features @ X_features.T

        # Compute error
        mean_error = np.mean(np.abs(kernel_matrix - kernel_matrix_approx))
        errors = np.append(errors, mean_error)

    # Plotting
    _, ax = plt.subplots(1, 1, figsize=(10, 6))
    font = {"fontname": "arial", "fontsize": 18}
    ax.plot(n_features, errors, color="deepskyblue", label=r"mean kernel error")
    ax.set_title("{} kernel error study".format(kernel_name), **font)
