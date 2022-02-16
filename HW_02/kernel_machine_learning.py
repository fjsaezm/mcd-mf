from typing import Callable, Tuple

import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import normalize

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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    X:
        Data matrix
    X_test:
        data matrix
    A:
        output variance
    ls:
        kernel lengthscale

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
    
    # Normalize the eigenvectors
    alpha_eigenvecs = normalize(alpha_eigenvecs, axis=0)
    
    # Compute the Gram matrix for the test observations
    K_test = kernel(X_test, X_test)
    
    # Compute the matrix K_tilda_test
    K_tilda_test = K_test - K_test @ ones_matrix / N \
            - ones_matrix @ K / N \
            + ones_matrix @ K @ ones_matrix / N**2
            
    # Project the test observations to the found principal components
    X_test_hat = K_tilda_test @ alpha_eigenvecs

    return X_test_hat, lambda_eigenvals, alpha_eigenvecs
