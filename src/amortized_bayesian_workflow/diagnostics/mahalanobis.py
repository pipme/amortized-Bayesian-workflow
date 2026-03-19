from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import EmpiricalCovariance


def _validate_train_summaries(rows: np.ndarray) -> np.ndarray:
    x = np.asarray(rows, dtype=float)
    if x.ndim != 2:
        raise ValueError(
            "Expected training summaries with shape (num_datasets, summary_dim)."
        )
    if x.shape[0] < 2:
        raise ValueError(
            "Need at least two training summaries to fit Mahalanobis reference."
        )
    return x


def _validate_summary(summary: np.ndarray, *, dim: int) -> np.ndarray:
    x = np.asarray(summary, dtype=float)
    if x.ndim != 1:
        raise ValueError("Expected summary with shape (summary_dim,).")
    if x.shape[0] != dim:
        raise ValueError(
            "Summary shape does not match fitted Mahalanobis reference shape."
        )
    return x


def _validate_alpha(alpha: float) -> float:
    a = float(alpha)
    if not (0.0 <= a <= 1.0):
        raise ValueError("Mahalanobis OOD alpha must be in [0, 1].")
    return a


@dataclass
class MahalanobisOODResult:
    statistic: float
    cutoff: float
    alpha: float
    rejected: bool


class MahalanobisReference:
    """Minimal Mahalanobis reference fitted on training summaries.

    This follows the simple workflow used in the reference notebook:
    1) fit covariance on training summaries,
    2) compute training Mahalanobis statistics,
    3) use a train quantile cutoff for OOD checks.
    """

    def __init__(
        self,
        *,
        cov: EmpiricalCovariance,
        train_statistics: np.ndarray,
    ) -> None:
        self.cov = cov
        self.train_statistics = np.asarray(train_statistics, dtype=float)

    @staticmethod
    def from_training_summaries(
        train_summaries: np.ndarray,
    ) -> "MahalanobisReference":
        x = _validate_train_summaries(train_summaries)
        cov = EmpiricalCovariance().fit(x)
        # sklearn returns squared Mahalanobis distance; we use the standard sqrt distance.
        train_statistics = np.sqrt(
            np.maximum(np.asarray(cov.mahalanobis(x), dtype=float), 0.0)
        )
        return MahalanobisReference(
            cov=cov,
            train_statistics=train_statistics,
        )

    @property
    def mean(self) -> np.ndarray:
        return np.asarray(self.cov.location_, dtype=float)

    @property
    def precision(self) -> np.ndarray:
        return np.asarray(self.cov.precision_, dtype=float)

    def threshold(self, *, alpha: float = 0.05) -> float:
        a = _validate_alpha(alpha)
        return float(np.quantile(self.train_statistics, 1.0 - a))

    def statistic(self, summary: np.ndarray) -> float:
        x = _validate_summary(summary, dim=self.mean.shape[0])
        # sklearn returns squared Mahalanobis distance; we use the standard sqrt distance.
        statistic = np.asarray(self.cov.mahalanobis(x[None, :]), dtype=float)
        return float(np.sqrt(np.maximum(statistic[0], 0.0)))

    def evaluate(
        self, summary: np.ndarray, *, alpha: float = 0.05
    ) -> MahalanobisOODResult:
        statistic = self.statistic(summary)
        cutoff = self.threshold(alpha=alpha)
        return MahalanobisOODResult(
            statistic=statistic,
            cutoff=cutoff,
            alpha=float(alpha),
            rejected=bool(statistic > cutoff),
        )
