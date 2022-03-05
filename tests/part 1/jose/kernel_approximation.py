"""
Authors:    alberto.suarez@uam.es
            joseantonio.alvarezo@estudiante.uam.es
"""

from __future__ import annotations

import warnings
from typing import Callable, Union, Type, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import datasets

from sklearn.gaussian_process.kernels import RBF as RBF_sklearn


class RandomFeaturesSampler(BaseEstimator, TransformerMixin):
    """Base class for random feature samplers."""

    def __init__(
        self,
        n_features_sampled: int,
        sampling_method: str,
    ) -> None:
        self.n_features_sampled = n_features_sampled
        self.sampling_method = sampling_method
        self.w = None

    def fit(
        self,
        X: np.ndarray,
        y: None = None,
    ) -> RandomFeaturesSampler:
        """
        Initialize w's for the random features.
        This should be implemented for each kernel.

        Each subclass should on their fir method should:
        1. Call this fir method.
        2. Fill the weights values self.w
        """
        self.w = None
        if self.sampling_method == "sin+cos":
            self._n_random_samples_w = self.n_features_sampled // 2
        elif self.sampling_method == "cos":
            self._n_random_samples_w = self.n_features_sampled
        else:
            raise ValueError("Please enter a correct sampling method")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Compute the random features.
        Assumes that the vector of w's has been initialized

        Parameters
        ----------
        X:
            Data matrix of shape (n_instances, n_features).

        Returns
        -------
        random_features:
            Array of shape (n_instances, n_features_sampled).
        """
        if self.w is None:
            raise ValueError("Use fit_transform to initialize w.")

        n_instances, n_features = np.shape(X)

        if np.shape(self.w)[1] != n_features:
            raise ValueError("Different # of features for X and w.")

        if self.sampling_method == "sin+cos":
            random_features = np.empty((n_instances, 2 * self._n_random_samples_w))
            random_features[:, ::2] = np.cos(X @ self.w.T)
            random_features[:, 1::2] = np.sin(X @ self.w.T)
            normalization_factor = np.sqrt(self._n_random_samples_w)

        elif self.sampling_method == "cos":
            """Q8. Implement the sampling method based
            on the second type of random features."""

            # Sample b from U[0, 2pi]
            b = 2 * np.pi * np.random.rand(self._n_random_samples_w)
            random_features = np.cos(X @ self.w.T + b)

            # Normalization factor obtained as explained in the notebook
            normalization_factor = np.sqrt(self._n_random_samples_w / 2)

        else:
            raise ValueError("Please enter a correct sampling method")

        random_features = random_features / normalization_factor

        return random_features

    def fit_transform(
        self, X: np.ndarray, X_prime: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Initialize  w's (fit) & compute random features (transform)."""
        self.fit(X)
        return self.transform(X)


class RandomFeaturesSamplerRBF(RandomFeaturesSampler):
    """Random Fourier Features for the RBF kernel."""

    def __init__(
        self,
        n_features_sampled: int = 100,
        sampling_method: str = "sin+cos",
        sigma: float = 1.0,
    ) -> None:
        super().__init__(n_features_sampled, sampling_method)
        self.sigma = sigma

    def fit(
        self,
        X: np.ndarray,
        y: None = None,
    ) -> RandomFeaturesSamplerRBF:
        """Initialize the w's for the random features."""
        super().fit(X)

        n_features = np.shape(X)[1]
        w_mean = np.zeros(n_features)

        # This implementation is NOT CONSISTENT witht the one in our
        # homework assigment, where we multiplied by sigma**2 instead
        # of dividing by it. This means the best sigmas will not be
        # the same after performing grid
        w_cov_matrix = np.identity(n_features) / self.sigma**2

        self.w = np.random.default_rng().multivariate_normal(
            w_mean,
            w_cov_matrix,
            self._n_random_samples_w,
        )

        return self


class RandomFeaturesSamplerMatern(RandomFeaturesSampler):
    """
    Random Fourier Features for the Matern kernel.

    The Fourier transform of the Matérn kernel is a
    Student's t distribution with twice the degrees of freedom.
    Ref. Chapter 4 of
    Carl Edward Rasmussen and Christopher K. I. Williams. 2005.
    Gaussian Processes for Machine Learning
    (Adaptive Computation and Machine Learning). The MIT Press.
    There is probably a mistake with the scale factor.
    """

    """
    Old constructor, commented instead of deleted in case is needed later
    def __init__(self, length_scale: float, nu: float) -> None:
    """

    def __init__(
        self,
        scale: float = 1.0,
        nu: float = 1.0,
        n_features_sampled: int = 100,
        sampling_method: str = "sin+cos",
    ) -> None:
        super().__init__(n_features_sampled, sampling_method)
        self.nu = nu
        self.scale = scale
        self.w = None

    def fit(
        self,
        X: np.ndarray,
        y: None = None,
    ) -> RandomFeaturesSamplerMatern:
        """Compute w's for random Matérn features."""
        # Scale of the Fourier tranform of the kernel
        super().fit(X)
        n_features = np.shape(X)[1]
        w_mean = np.zeros(n_features)
        w_cov_matrix = np.identity(n_features) / self.scale**2

        self.w = random_multivariate_student_t(
            w_mean,
            w_cov_matrix,
            2.0 * self.nu,
            self._n_random_samples_w,
        )
        return self


def random_multivariate_student_t(
    mean: np.ndarray,
    cov_matrix: np.ndarray,
    degrees_of_freedom: float,
    n_samples: int,
) -> np.ndarray:
    """
    Generate samples from a multivariate Student's t.
    https://en.wikipedia.org/wiki/Multivariate_t-distribution#Definition
    """

    # Dimensions of multivariate Student's t distribution.
    D = len(mean)

    rng = np.random.default_rng()
    x = rng.chisquare(degrees_of_freedom, n_samples) / degrees_of_freedom

    Z = rng.multivariate_normal(
        np.zeros(D),
        cov_matrix,
        n_samples,
    )

    X = mean + Z / np.sqrt(x)[:, np.newaxis]
    return X


def random_multivariate_cauchy(sample_shape, gamma, x_0):
    """
        Obtains n_features_sampled samples following a Cauchy distribution
        with parameters gamma and x_0.
        https://en.wikipedia.org/wiki/Cauchy_distribution

    Args:
        n_features_sampled (int): Number of features to be sampled.
        gamma (float): scale
        x_0 (float): center

    Return:
        List of random features sampled.
    """

    def cauchy_inverse_cdf(x, gamma, x_0):
        return x_0 + gamma * np.tan(np.pi * (x - 0.5))

    # U ~ U[0, 1]
    U = np.random.rand(*sample_shape)
    return cauchy_inverse_cdf(U, gamma, x_0)


class RandomFeaturesSamplerExp(RandomFeaturesSampler):
    """Random Fourier Features for the exponential kernel."""

    def __init__(
        self,
        n_features_sampled: int = 100,
        sampling_method: str = "sin+cos",
        length_scale_kernel: float = 1,
    ) -> None:
        super().__init__(n_features_sampled, sampling_method)
        # gamma
        self.length_scale_kernel = length_scale_kernel

    def fit(
        self,
        X: np.ndarray,
        y: None = None,
    ) -> RandomFeaturesSamplerExp:
        """Initialize the w's for the random features."""
        super().fit(X)

        """
            Q6. Write code to generate random Fourier features
            corresponding to the exponential kernel in D dimensions.
        """
        samples_dimension = np.shape(X)[1]
        sample_shape = (self._n_random_samples_w, samples_dimension)

        # The Cauchy's gamma is the inverse of our gamma
        cauchy_scale = 1.0 / self.length_scale_kernel

        self.w = random_multivariate_cauchy(sample_shape, gamma=cauchy_scale, x_0=0)

        return self


class NystroemFeaturesSampler(BaseEstimator, TransformerMixin):
    """Sample Nystroem features."""

    def __init__(
        self,
        n_features_sampled: int = 100,
        kernel: Callable[[np.ndarray, np.ndarray], np.ndarray] = RBF_sklearn(),
    ) -> None:
        # _kernel -> kernel due to SKlearn compatibility
        self.kernel = kernel
        self.component_indices_ = None

        # J
        self._X_reduced = None

        # W
        self._reduced_kernel_matrix = None

        # (W+)^1/2
        self._sqrtm_pinv_reduced_kernel_matrix = None

        # Needed for compatibility
        self.n_features_sampled = n_features_sampled

    def fit(
        self,
        X: np.ndarray,
        y: None = None,
    ) -> NystroemFeaturesSampler:
        """Precompute auxiliary quantities for Nystroem features."""
        n_instances = len(X)

        # Sample subset of training instances.
        rng = np.random.default_rng()

        # Check dimensions to avoid selecting more features than available
        n_features_to_sample = min(self.n_features_sampled, n_instances)

        self.component_indices_ = rng.choice(
            range(n_instances),
            size=n_features_to_sample,
            replace=False,
        )

        self._X_reduced = X[self.component_indices_, :]

        # Compute reduced kernel matrix.
        self._reduced_kernel_matrix = self.kernel(self._X_reduced, self._X_reduced)
        self._reduced_kernel_matrix = (
            self._reduced_kernel_matrix + self._reduced_kernel_matrix.T
        ) / 2.0  # enforce symmetry of kernel matrix

        # Compute auxiliary quantities.
        self._sqrtm_pinv_reduced_kernel_matrix = sp.linalg.sqrtm(
            np.linalg.pinv(self._reduced_kernel_matrix, rcond=1.0e-6, hermitian=True)
        )

        # Check that complex part is negligible and eliminate it
        if np.iscomplexobj(self._sqrtm_pinv_reduced_kernel_matrix):
            threshold_imaginary_part = 1.0e-6
            max_imaginary_part = np.max(
                np.abs(np.imag(self._sqrtm_pinv_reduced_kernel_matrix))
            )
            if max_imaginary_part > threshold_imaginary_part:
                warnings.warn("Maximum imaginary part is {}".format(max_imaginary_part))

            self._sqrtm_pinv_reduced_kernel_matrix = np.real(
                self._sqrtm_pinv_reduced_kernel_matrix
            )

        return self

    def approximate_kernel_matrix(
        self,
        X: np.ndarray,
        n_features_sampled: int,
        X_prime: Optional[np.ndarray] = None,  # Needed for sklearn compatibility
    ) -> np.ndarray:
        """Approximate the kernel matrix using Nystroem features."""
        X_features = self.fit_transform(X)
        return X_features @ X_features.T

    def transform(self, X_prime: np.ndarray) -> np.ndarray:
        """Compute Nystroem features with precomputed quantities."""
        J = self.kernel(X_prime, self._X_reduced)
        return J @ self._sqrtm_pinv_reduced_kernel_matrix

    def fit_transform(
        self,
        X: np.ndarray,
        X_prime: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # Compute Nystrom features.
        self.fit(X)

        # if X_prime is None:
        #    X_prime = X

        return self.transform(X)


def demo_kernel_approximation_features(
    X: np.ndarray,
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    features_sampler_class: Union[
        Type[RandomFeaturesSampler], Type[NystroemFeaturesSampler]
    ],
    features_sampler_kwargs: np.ndarray,
    n_features: np.array,
    sampler_name=None,
) -> None:
    """Kernel approximation using random sampled features.
    Either RFF or Nyström features."""
    n_plots = len(n_features) + 1
    fig, axes = plt.subplots(1, n_plots)
    fig.set_size_inches(15, 4)
    font = {"fontname": "arial", "fontsize": 18}

    kernel_matrix = kernel(X, X)
    axes[0].imshow(kernel_matrix, cmap=plt.cm.Blues)
    axes[0].set_title("Exact kernel", **font)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    for n_f, ax in zip(n_features, axes[1:]):
        features_sampler = features_sampler_class(
            n_features_sampled=n_f, **features_sampler_kwargs
        )

        X_features = features_sampler.fit_transform(X)
        kernel_matrix_approx = X_features @ X_features.T

        ax.imshow(kernel_matrix_approx, cmap=plt.cm.Blues)

        err_approx = kernel_matrix - kernel_matrix_approx
        err_mean = np.mean(np.abs(err_approx))
        err_max = np.max(np.abs(err_approx))

        ax.set_xlabel(
            "err (mean) = {:.4f} \n err (max) = {:.4f}".format(err_mean, err_max),
            **font,
        )

        ax.set_title("{} features".format(n_f), **font)

        ax.set_xticks([])
        ax.set_yticks([])

    if sampler_name is not None:
        plt.suptitle("{} kernel approximation".format(sampler_name), **font)
    plt.tight_layout()
    plt.show()


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


"""
1. Non-linear SVM + RBF kernel [C, gamma]
2. Linear SVM + RBF random features [C, gamma, n_features]
3. Linear SVM + RBF Nyström features [C, gamma, n_features]
4. Non-linear SVM + exponential kernel [C, length_scale]
5. Linear SVM + exponential random features [C, length_scale, n_features]
6. Linear SVM + exponential Nyström features [C, length_scale, n_features]
"""
