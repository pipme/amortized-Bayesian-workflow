import numpy as np


def nested_rhat(
    ary: np.ndarray,
    num_superchains: int | None = None,
    superchains_id: np.ndarray | None = None,
):
    """Compute the nested R-hat.

    Parameters
    ----------
        ary: A numpy array of shape (num_chains, num_samples, num_values)
    """
    if ary.ndim == 2:
        R_hat_nu_values = _nested_rhat(ary, num_superchains, superchains_id)
    elif ary.ndim == 3:
        R_hat_nu_values = np.zeros(ary.shape[2])
        for i in range(ary.shape[2]):
            ary_i = ary[:, :, i]
            R_hat_nu = _nested_rhat(ary_i, num_superchains, superchains_id)
            R_hat_nu_values[i] = R_hat_nu
    else:
        raise ValueError("The input array should have 2 or 3 dimensions.")
    return R_hat_nu_values


def _nested_rhat(
    ary: np.ndarray,
    num_superchains: int | None = None,
    superchains_id: np.ndarray | None = None,
):
    """Compute the nested R-hat.

    Parameters
    ----------
        ary: A numpy array of shape (num_chains, num_samples)
        num_superchains: The number of superchains. If provided, the chains will be split into superchains based on this number, in sequential order.
        superchains_id: A numpy array of shape (num_chains,) containing the superchain id of each chain.
    """
    # Only one of superchains_id or num_superchains should be provided
    assert (superchains_id is not None) != (num_superchains is not None)

    if num_superchains is not None:
        # Split the chains into superchains, check if the number of chains is divisible by num_superchains
        assert ary.shape[0] % num_superchains == 0

        superchains_id = np.repeat(
            np.arange(num_superchains), ary.shape[0] // num_superchains
        )

    assert ary.ndim == 2
    assert superchains_id.ndim == 1
    assert len(superchains_id) == ary.shape[0]
    # Either num_superchains or superchains_id should be provided
    superchains_id_unique, counts = np.unique(
        superchains_id, return_counts=True
    )
    # Each superchain should have the same number of subchains
    assert counts.min() == counts.max()

    num_superchains = len(superchains_id_unique)
    num_chains = ary.shape[0]

    K = num_superchains
    M = num_chains // K
    N = ary.shape[1]
    assert N > 1 or M > 1, f"Got {N=}, {M=}"

    # Reorder array shape to (num_samples, num_chains_per_superchain, num_superchains), i.e., (N, M, K)
    reorder_inds = np.argsort(superchains_id)
    ary = ary[reorder_inds]
    ary = ary.T  # (N, K * M)
    ary = np.reshape(ary, (N, M, K), order="F")  # (N, M, K)
    f_bar_nmk = ary

    # sample mean of each superchain
    f_bar_ddk = ary.mean(axis=(0, 1))  # (K,)

    # sample mean of all chains
    f_bar = f_bar_ddk.mean()

    f_bar_dmk = ary.mean(axis=0)  # (M, K)

    # estimator of the between-superchain variance
    B_hat_nu = 1 / (K - 1) * np.sum((f_bar_ddk - f_bar) ** 2)
    # estimator of the between-chain variance
    if M > 1:
        B_tilde_k = (
            1 / (M - 1) * np.sum((f_bar_dmk - f_bar_ddk) ** 2, axis=0)
        )  # (K,)
    elif M == 1:
        B_tilde_k = 0
    else:
        raise ValueError("M should be greater than 0.")

    # estimator of the within-chain variance
    if N > 1:
        W_tilde_k = (
            1 / (N - 1) * np.sum((f_bar_nmk - f_bar_dmk) ** 2, axis=0)
        )  # (M, K)
        W_tilde_k = 1 / M * np.sum(W_tilde_k, axis=0)  # (K,)
    elif N == 1:
        W_tilde_k = 0
    else:
        raise ValueError("N should be greater than 0.")

    # estimator of the within-superchain variance
    W_hat_nu = 1 / K * np.sum(B_tilde_k + W_tilde_k)

    R_hat_nu = np.sqrt(1 + B_hat_nu / W_hat_nu)
    return R_hat_nu
