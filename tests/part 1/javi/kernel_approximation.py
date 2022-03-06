from __future__ import annotations

import warnings
from typing import Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import datasets


class RandomFeaturesSampler(BaseEstimator, TransformerMixin):
    """Base class for random feature samplers."""

    def __init__(
        self,
        n_features_sampled: int,
        sampling_method: str,
    ) -> None:
        self.n_features_sampled = n_features_sampled
        self.sampling_method = sampling_method

    def fit(
        self,
        X: np.ndarray,
        y: None = None,
    ) -> RandomFeaturesSampler:
        """Initialize w's for the random features.
        This should be implemented for each kernel."""
        self.w = None
        if self.sampling_method == 'sin+cos':
            self._n_random_samples_w = self.n_features_sampled // 2
        elif self.sampling_method == 'cos':
            self._n_random_samples_w = self.n_features_sampled
        else:
            raise ValueError('Please enter a correct sampling method')

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
        if (self.w is None):
            raise ValueError('Use fit_transform to initialize w.')

        n_instances, n_features = np.shape(X)

        if (np.shape(self.w)[1] != n_features):
            raise ValueError('Different # of features for X and w.')

        if self.sampling_method == 'sin+cos':
            random_features = np.empty(
                (n_instances, 2 * self._n_random_samples_w)
            )
            random_features[:, ::2] = np.cos(X @ self.w.T)
            random_features[:, 1::2] = np.sin(X @ self.w.T)
            normalization_factor = np.sqrt(self._n_random_samples_w)

        elif self.sampling_method == 'cos':
            b = np.random.default_rng().uniform( low = 0.0,
                                                high = 2.0*np.pi,
                                                size = self._n_random_samples_w )
            random_features = np.cos( X @ self.w.T + b )
            normalization_factor = 1.0/np.sqrt(2.0/self._n_random_samples_w)
                                     
        else:
            raise ValueError('Please enter a correct sampling method')

        random_features = random_features / normalization_factor

        return random_features


class RandomFeaturesSamplerRBF(RandomFeaturesSampler):
    """ Random Fourier Features for the RBF kernel. """

    def __init__(
        self,
        n_features_sampled: int = 100,
        sampling_method: str = 'sin+cos',
        sigma_kernel: float = 2.0
    ) -> None:
        super().__init__(n_features_sampled, sampling_method)
        self.sigma_kernel = sigma_kernel

    def fit(
        self,
        X: np.ndarray,
        y: None = None,
    ) -> RandomFeaturesSamplerRBF:
        """Initialize the w's for the random features."""
        super().fit(X)

        n_features = np.shape(X)[1]
        w_mean = np.zeros(n_features)
        w_cov_matrix = np.identity(n_features) / self.sigma_kernel**2

        self.w = np.random.default_rng().multivariate_normal(
            w_mean,
            w_cov_matrix,
            self._n_random_samples_w,
        )
        return self


class RandomFeaturesSamplerMatern(RandomFeaturesSampler):
    """Random Fourier Features for the Matern kernel."""

    """
    The Fourier transform of the Matérn kernel is a
    Student's t distribution with twice the degrees of freedom.
    Ref. Chapter 4 of
    Carl Edward Rasmussen and Christopher K. I. Williams. 2005.
    Gaussian Processes for Machine Learning
    (Adaptive Computation and Machine Learning). The MIT Press.
    There is probably a mistake with the scale factor.
    """

    def __init__(
        self,
        n_features_sampled: int,
        sampling_method: str,
        length_scale_kernel: float,
        nu_matern_kernel: float
    ) -> None:
        super().__init__(n_features_sampled, sampling_method)
        self.nu_matern_kernel = nu_matern_kernel
        self.length_scale_kernel = length_scale_kernel

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
        w_cov_matrix = np.identity(n_features) / self.length_scale_kernel**2

        self.w = random_multivariate_student_t(
            w_mean,
            w_cov_matrix,
            2.0 * self.nu_matern_kernel,
            self._n_random_samples_w,
        )
        return self


def random_multivariate_student_t(
    mean: np.ndarray,
    cov_matrix: np.ndarray,
    degrees_of_freedom: float,
    n_samples: int,
) -> np.ndarray:
    """Generate samples from a multivariate Student's t.
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



class RandomFeaturesSamplerExp(RandomFeaturesSampler):
    """ Random Fourier Features for the exponential kernel. """

    def __init__(
        self,
        n_features_sampled: int = 100,
        sampling_method: str = 'sin+cos',
        length_scale_kernel: float = 2.0
    ) -> None:
        super().__init__(n_features_sampled, sampling_method)
        self.length_scale_kernel = length_scale_kernel

    def fit(
        self,
        X: np.ndarray,
        y: None = None,
    ) -> RandomFeaturesSamplerExp:
        """Initialize the w's for the random features."""
        super().fit(X)

        # Declare the inverse cdf needed for inverse sampling method
        cauchy_inverse_cdf = lambda p,gamma : (1/gamma)* np.tan(np.pi*(p-0.5))

        rng = np.random.default_rng()
        U = rng.random((self._n_random_samples_w, X.shape[1]))
        self.w = cauchy_inverse_cdf(U, self.length_scale_kernel)

        return self

class NystroemFeaturesSampler(BaseEstimator, TransformerMixin):
    """Sample Nystroem features. """

    def __init__(
        self,
        n_features_sampled: int,
        kernel: Callable[[np.ndarray, np.ndarray], np.ndarray]

    ) -> None:
        self.n_features_sampled=n_features_sampled
        self._kernel = kernel

    def fit(
        self,
        X: np.ndarray,
        y: None = None,
    ) -> NystroemFeaturesSampler:
        """Precompute auxiliary quantities for Nystroem features."""
        n_instances = len(X)
        # Sample subset of training instances.
        rng = np.random.default_rng()
        self.component_indices_ = rng.choice(
            range(n_instances),
            size=self.n_features_sampled,
            replace=False,
        )

        self._X_reduced = X[self.component_indices_, :]

        # Compute reduced kernel matrix.
        self._reduced_kernel_matrix = self._kernel(
            self._X_reduced,
            self._X_reduced
        )

        self._reduced_kernel_matrix = (
            self._reduced_kernel_matrix + self._reduced_kernel_matrix.T
        ) / 2.0  # enforce symmetry of kernel matrix

        # Compute auxiliary quantities.
        self._sqrtm_pinv_reduced_kernel_matrix = sp.linalg.sqrtm(
            np.linalg.pinv(
                self._reduced_kernel_matrix,
                rcond=1.0e-6,
                hermitian=True
            )
        )

        # Check that complex part is negligible and eliminate it
        if np.iscomplexobj(self._sqrtm_pinv_reduced_kernel_matrix):
            threshold_imaginary_part = 1.0e-6
            max_imaginary_part = np.max(
                np.abs(np.imag(self._sqrtm_pinv_reduced_kernel_matrix))
            )
            if max_imaginary_part > threshold_imaginary_part:
                warnings.warn(
                    'Maximum imaginary part is {}'.format(max_imaginary_part)
                )

            self._sqrtm_pinv_reduced_kernel_matrix = np.real(
                self._sqrtm_pinv_reduced_kernel_matrix
            )

        return self

    def approximate_kernel_matrix(
        self,
        X: np.ndarray,
        n_features_sampled: int
    ) -> np.ndarray:
        """Approximate the kernel matrix using Nystroem features."""
        X_nystroem = self.fit_transform(n_features_sampled, X)
        kernel_matrix_approx = X_nystroem @ X_nystroem.T
        return kernel_matrix_approx

    """ def fit_transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        # Compute Nystrom features.
        self.fit(X)
        if X_prime is None:
            X_prime = X
        X_prime_nystroem = self.transform(X_prime)
        return X_prime_nystroem
    """

    def transform(self, X_prime: np.ndarray) -> np.ndarray:
        """Compute Nystroem features with precomputed quantities."""
        reduced_kernel_matrix_columns = self._kernel(X_prime, self._X_reduced)

        X_prime_nystroem = (
            reduced_kernel_matrix_columns
            @ self._sqrtm_pinv_reduced_kernel_matrix
        )

        return X_prime_nystroem


def demo_kernel_approximation_features(
    X: np.ndarray,
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    features_samplers: List[Union[Rlength_scale_kernelandomFeaturesSampler, NystroemFeaturesSampler]],
) -> None:
    """Kernel approximation using random sampled features.
    Either RFF or Nyström features."""
    n_plots = len(features_samplers) + 1
    fig, axes = plt.subplots(1, n_plots)
    fig.set_size_inches(15, 4)
    font = {'fontname': 'arial', 'fontsize': 18}

    kernel_matrix = kernel(X, X)
    axes[0].imshow(kernel_matrix, cmap=plt.cm.Blues)
    axes[0].set_title('Exact kernel', **font)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    for features_sampler, ax in zip(features_samplers, axes[1:]):

        X_features = features_sampler.fit_transform(X)

        kernel_matrix_approx = X_features @ X_features.T

        ax.imshow(kernel_matrix_approx, cmap=plt.cm.Blues)

        err_approx = kernel_matrix - kernel_matrix_approx
        err_mean = np.mean(np.abs(err_approx))
        err_max = np.max(np.abs(err_approx))

        ax.set_xlabel('err (mean) = {:.4f} \n err (max) = {:.4f}'.format(
            err_mean,
            err_max
        ), **font)

        ax.set_title(
            '{} features'.format(features_sampler.n_features_sampled),
            **font,
        )

        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
    plt.show()


def generate_curve_dataset(n_instances = 1000):
    """
    Generates an 3-dimensional S given a number of instances
    """
    X,t = datasets.make_s_curve(n_instances, noise = 0.1)
    X = X[np.argsort(t)]
   
    return X,t


def plot_curve_dataset(data,t):

    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[:,0], data[:,1],data[:,2], c = t, cmap = plt.cm.Spectral)
    ax.view_init(10,80)
    plt.show()


def plot_kernel_approximation_error(
    X: np.ndarray,
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    features_samplers: List[Union[Rlength_scale_kernelandomFeaturesSampler, NystroemFeaturesSampler]],
    n_features : np.ndarray,
    error_function : Callable[[np.array]]
    ) -> None:
    """
    Kernel approximation using random sampled features.
    Either RFF or Nyström features.
    """

    kernel_matrix = kernel(X, X)

    means = []
    for features_sampler  in features_samplers:
        X_features = features_sampler.fit_transform(X)
        kernel_matrix_approx = X_features @ X_features.T
        err_approx = kernel_matrix - kernel_matrix_approx
        means.append(np.mean(np.abs(err_approx)))

    values = error_function(n_features)
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot()
    ax.plot(n_features, means, color= "tomato", label="Mean error")
    ax.plot(n_features, values,color = "dodgerblue", label = "$1/\sqrt{n}$")
    ax.legend()
    ax.set_title("Comparison of estimated MC error and theoretical MC error")
    plt.show()

    
