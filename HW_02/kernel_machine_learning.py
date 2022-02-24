from typing import Callable, Tuple

import numpy as np
from scipy.spatial import distance

from sklearn.preprocessing import normalize
from sklearn.utils.extmath import svd_flip

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as anim

def linear_kernel(
    X: np.ndarray,
    X_prime: np.ndarray,
) -> np.ndarray:
    return X @ X_prime.T


def exponential_kernel(
    X: np.ndarray,
    X_prime: np.ndarray,
    A: float,
    l: float
) -> np.ndarray:
    d = distance.cdist(X, X_prime, metric='minkovski', p=1.0)
    return A * np.exp(- d / l)


def rbf_kernel(
    X: np.ndarray,
    X_prime: np.ndarray,
    A: float,
    ls: float,
) -> np.ndarray:
    """
    Parameters
    ----------
    X:
        Data matrix
    X_prime:
        Data matrix
    A:
        Output variance
    ls:
        Kernel lengthscale

    Returns
    -------
    kernel matrix

    Notes
    -------
    Alternative parametrization (e.g. en sklearn)
    gamma = 0.5 / ls**2

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import gaussian_process_regression as gp
    >>> X = np.array([[1,2], [3, 4], [5,6]])
    >>> X_prime = np.array([[1,2], [3, 4]])
    >>> A, l = 3, 10.0
    >>> kernel_matrix = gp.rbf_kernel(X, X_prime, A, l)
    >>> print(kernel_matrix)
    """
    d = distance.cdist(X, X_prime, metric='euclidean')
    return A * np.exp(-0.5 * (d / ls)**2)


def kernel_pca(
    X: np.ndarray,
    X_test: np.ndarray,
    kernel: Callable,
    deterministic_orientation: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    X:
        Data matrix
    X_test:
        data matrix
    kernel:
        Kernel function
    deterministic_orientation:
        whether or not to orientate the resulsts
        as done by Sklearn

    Returns
    -------
    X_test_hat:
        Projection of X_test on the principal components
    lambda_eigenvals:
        Eigenvalues of the centered kernel
    alpha_eigenvecs:
        Principal components. These are the eigenvectors
        of the centered kernel with the RKHS normalization

    Notes
    -------
    In the corresponding method of sklearn the eigenvectors
    are normalized in l2.

    """
    
    # Create NxN ones matrix
    N = len(X)
    ones_matrix = 1.0 * np.ones((N, N))
    
    # Compute the Gram matrix for the train observations
    K = kernel(X, X)
    
    # Compute the Gram matrix of the centered kernel
    K_tilda = K - K @ ones_matrix / N - ones_matrix @ K / N \
            + ones_matrix @ K @ ones_matrix / N**2
    
    # Obtain the SVD decomposition of the kernel matrix
    alpha_eigenvecs, lambda_eigenvals, V = np.linalg.svd(K_tilda, full_matrices=True)
    
    # Make the eigenvector signs match the ones obtained from Sklearn
    if deterministic_orientation:
        alpha_eigenvecs, _ = svd_flip(alpha_eigenvecs, V)
    
    # Normalize the eigenvectors so their norm equals 1/lambda_i
    for i, lambda_i in enumerate(np.sqrt(lambda_eigenvals)):
        alpha_eigenvecs[:,i] /= lambda_i
    
    # Compute the Gram matrix for the test observations
    K_test = kernel(X_test, X_test)
    
    # Compute the matrix K_tilda_test
    K_tilda_test = K_test - K_test @ ones_matrix / N \
            - ones_matrix @ K / N \
            + ones_matrix @ K @ ones_matrix / N**2
            
    # Project the test observations to the found principal components
    X_test_hat = K_tilda_test @ alpha_eigenvecs
    
    return X_test_hat, lambda_eigenvals, alpha_eigenvecs


class AnimationKPCA:
    
    def __init__(
        self,
        xlims: Tuple[float, float] = (-1, 1),
        ylims: Tuple[float, float] = (-1, 1),
        n_frames: int = 50,
    ):
        self.xlims = xlims
        self.ylims = ylims
        self.n_frames = n_frames
        self.gammas = 2*np.logspace(-3, 4, n_frames)
        self.A = 1.0
        self.L = 1.0
        
    def _init_frame_plot(
        self,
        ax: matplotlib.axes.Axes,
        gamma: float,
    ):
        ax.clear()
        ax.set_title(r"Projection by KPCA ($\gamma=${:.3f})".format(gamma))
        ax.set_xlabel(r"1st principal component in space induced by $\phi$")
        ax.set_ylabel("2nd principal component")
        ax.set_xlim(self.xlims)
        ax.set_ylim(self.ylims)
    
    def _update_animation(
        self,
        i_frame: int,
        ax: matplotlib.axes.Axes,
        X: np.ndarray,
        X_test: np.ndarray,
        reds: np.ndarray,
        blues: np.ndarray,
    ):
        gamma = self.gammas[i_frame]
        self.L = np.sqrt(0.5/gamma)
        self._init_frame_plot(ax, gamma)

        X_kpca, _, _ = kernel_pca(X, X_test, self.kernel,
                                  deterministic_orientation=True)

        ax.scatter(X_kpca[reds, 0], X_kpca[reds, 1], c="red",
                   s=20, edgecolor='k')
        ax.scatter(X_kpca[blues, 0], X_kpca[blues, 1], c="blue",
                   s=20, edgecolor='k')
    
    def animate(
        self,
        X: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> matplotlib.animation.FuncAnimation:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)

        def kernel(X, X_prime):
            return rbf_kernel(X, X_prime, self.A, self.L)

        self.kernel = kernel
        reds = y_test == 0
        blues = y_test == 1

        return anim.FuncAnimation(
            fig,
            self._update_animation,
            frames=self.n_frames,
            repeat=False,
            fargs=(ax, X, X_test, reds, blues,),
            blit = False
        )
    
    
def plot_KPCA(gamma, X, y, A=1.0, lims=None):
    # Parameters
    L = np.sqrt(0.5/gamma)
    reds = y == 0
    blues = y == 1

    # Kernel definition
    def local_rbf_kernel(X, X_prime):
        return rbf_kernel(X, X_prime, A, L)

    # PCA
    X_kpca, _, _ = \
        kernel_pca(X, X, local_rbf_kernel,
                   deterministic_orientation=True)

    # Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)

    ax.scatter(X_kpca[reds, 0], X_kpca[reds, 1], c="red",
                s=20, edgecolor='k')
    ax.scatter(X_kpca[blues, 0], X_kpca[blues, 1], c="blue",
                s=20, edgecolor='k')

    if lims is not None:
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    ax.set_title("Projection by KPCA (gamma={})".format(gamma))
    ax.set_xlabel(r"1st principal component in space induced by $\phi$")
    ax.set_ylabel("2nd component")
    plt.show()
