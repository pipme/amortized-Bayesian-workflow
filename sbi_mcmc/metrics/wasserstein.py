import numpy as np
import ot


def wasserstein_distance(X1, X2, p=1, num_samples=1000):
    """Compute Wasserstein distance between two distributions.

    Parameters
    ----------
    X1 : array-like, shape (n_samples, n_dims)
        Empirical samples from the first distribution.
    X2 : array-like, shape (n_samples, n_dims)
        Empirical samples from the second distribution.
    p : int, default=1
        The p-th Wasserstein distance to compute.
    num_samples : int, default=1000
        The number of samples to use computing the Wasserstein distance.

    Returns
    -------
    float
        Wasserstein distance between the two empirical distributions.
    """
    assert X1.shape[0] >= num_samples
    assert X2.shape[0] >= num_samples
    X1_thin = X1[:: X1.shape[0] // num_samples]
    X2_thin = X2[:: X2.shape[0] // num_samples]
    M = ot.dist(X1_thin, X2_thin, metric="euclidean") ** p
    weights_1 = np.ones(X1_thin.shape[0]) / X1_thin.shape[0]
    weights_2 = np.ones(X2_thin.shape[0]) / X2_thin.shape[0]
    return ot.emd2(weights_1, weights_2, M) ** (1 / p)
