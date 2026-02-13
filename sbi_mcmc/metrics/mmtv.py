import numpy as np
from scipy.integrate import quad, simpson
from scipy.interpolate import interp1d

from .stats import kde1d


def mtv(
    X1: np.ndarray | callable | None = None,
    X2: np.ndarray | callable | None = None,
    posterior=None,
    *args,
    **kwargs,
) -> np.ndarray | Exception:
    """
    Marginal total variation distances between two sets of posterior samples.

    Compute the total variation distance between posterior samples X1 and
    posterior samples X2, separately for each dimension (hence
    "marginal" total variation distance, MTV).

    Parameters
    ----------
    X1 : np.ndarray or callable, optional
        A ``N1``-by-``D`` matrix of samples, typically N1 = 1e5.
        Alternatively, may be a callable ``X1(x, d)`` which returns the marginal
        pdf along dimension ``d`` at point(s) ``x``.
    X2 : np.ndarray or callable, optional
        Another ``N2``-by-``D`` matrix of samples, typically N2 = 1e5.
        Alternatively, may be a callable ``X2(x, d)`` which returns the marginal
        pdf along dimension ``d`` at point(s) ``x``.

    Returns
    -------
    mtv: np.ndarray
        A ``D``-element vector whose elements are the total variation distance
        between the marginal distributions of ``vp`` and ``vp1`` or ``samples``,
        for each coordinate dimension.

    Raises
    ------
    ValueError
        Raised if neither ``vp2`` nor ``samples`` are specified.

    Notes
    -----
    The total variation distance between two densities `p1` and `p2` is:

    .. math:: TV(p1, p2) = \\frac{1}{2} \\int | p1(x) - p2(x) | dx.

    """
    # If samples are not provided, fetch them from the posterior object:
    if all(a is None for a in [X1, X2]):
        raise ValueError("No samples/callable or posterior provided.")

    D = X1.shape[1]

    nkde = 2**13
    mtv = np.zeros((D,))

    def f1(x, xmesh, yy):
        return interp1d(
            xmesh,
            yy,
            kind="cubic",
            fill_value=np.array([0]),
            bounds_error=False,
        )(x)

    # Compute marginal total variation
    for d in range(D):
        if not callable(X1):
            if kwargs.get("kde") == "FFTKDE":
                from KDEpy import FFTKDE

                # FFTKDE in KDEpy is more reliable when there are many repeated samples
                x1mesh, yy1 = (
                    FFTKDE(bw="silverman").fit(X1[:, d]).evaluate(nkde)
                )
            else:
                yy1, x1mesh, _ = kde1d(X1[:, d], nkde)
            # Ensure normalization
            yy1 = yy1 / simpson(yy1, x1mesh)
        else:
            raise ValueError("Callable not supported yet.")

        if not callable(X2):
            if kwargs.get("kde") == "FFTKDE":
                x2mesh, yy2 = (
                    FFTKDE(bw="silverman").fit(X2[:, d]).evaluate(nkde)
                )
            else:
                yy2, x2mesh, _ = kde1d(X2[:, d], nkde)
            # Ensure normalization
            yy2 = yy2 / simpson(yy2, x2mesh)

        else:
            raise ValueError("Callable not supported yet.")

        def f(x, x1mesh=x1mesh, yy1=yy1, x2mesh=x2mesh, yy2=yy2):
            return np.abs(f1(x, x1mesh, yy1) - f1(x, x2mesh, yy2))

        lb = min(x1mesh[0], x2mesh[0])
        ub = max(x1mesh[-1], x2mesh[-1])
        if not np.isinf(lb) and not np.isinf(ub):
            # Grid integration (faster)
            grid = np.linspace(lb, ub, int(1e6))
            y_tot = f(grid)
            mtv[d] = 0.5 * simpson(y_tot, grid)
        else:
            # QUADPACK integration (slower)
            mtv[d] = 0.5 * quad(f, lb, ub)[0]
    mtv = np.maximum(0, mtv)  # Ensure non-negative
    mtv = np.minimum(1, mtv)  # Ensure bounded by 1
    return mtv


def mmtv(
    X1: np.ndarray | callable | None = None,
    X2: np.ndarray | callable | None = None,
    posterior=None,
    *args,
    **kwargs,
) -> float | Exception:
    """
    Mean marginal total variation dist. between two set of posterior samples.
    """
    result = mtv(X1, X2, posterior)
    if isinstance(result, Exception):
        return result
    else:
        return result.mean()
