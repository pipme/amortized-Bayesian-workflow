from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class DatasetResult:
    dataset_id: int
    status: str
    message: str
    posterior_draws: np.ndarray | None = None
    amortized_draws: np.ndarray | None = None
    amortized: Mapping[str, Any] = field(default_factory=dict)
    psis: Mapping[str, Any] = field(default_factory=dict)
    mcmc: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    def suggestion(self) -> str | None:
        if self.status == "success":
            return None
        if self.status == "needs_review":
            return (
                "Increase warmup, switch MCMC backend/sampler, or inspect model "
                "log density for this dataset."
            )
        if self.status == "failed":
            return (
                "Check simulator/likelihood consistency, initialization quality, "
                "and numerical stability (NaN/-inf log posterior values)."
            )
        return None


@dataclass(frozen=True)
class WorkflowReport:
    results: tuple[DatasetResult, ...]
    config: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def summary_table(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for r in self.results:
            rows.append(
                {
                    "dataset_id": r.dataset_id,
                    "status": r.status,
                    "message": r.message,
                    "mahalanobis_statistic": r.amortized.get(
                        "mahalanobis_statistic",
                        r.amortized.get("mahalanobis_distance"),
                    ),
                    "mahalanobis_cutoff": r.amortized.get(
                        "mahalanobis_cutoff",
                        r.amortized.get("mahalanobis_threshold"),
                    ),
                    "ood_rejected": r.amortized.get("ood_rejected"),
                    "pareto_k": r.psis.get("pareto_k"),
                    "psis_ess": r.psis.get("ess"),
                    "mcmc_backend": r.mcmc.get("backend"),
                    "mcmc_rhat": r.mcmc.get("rhat", r.mcmc.get("nested_rhat")),
                    "error": r.error,
                }
            )
        return rows

    def failed_datasets(self) -> list[DatasetResult]:
        return [r for r in self.results if r.status != "success"]

    def posterior_draws(self, dataset_id: int) -> np.ndarray:
        for r in self.results:
            if r.dataset_id == dataset_id:
                if r.posterior_draws is None:
                    raise ValueError(
                        f"No posterior draws stored for dataset {dataset_id}."
                    )
                return r.posterior_draws
        raise KeyError(dataset_id)

    def collect_posterior_draws(self) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        for r in self.results:
            if r.posterior_draws is not None:
                out[r.dataset_id] = r.posterior_draws
        return out

    def plot_diagnostics(self, dataset_id: int):
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "matplotlib is required for plotting diagnostics."
            ) from exc

        row = None
        for r in self.results:
            if r.dataset_id == dataset_id:
                row = r
                break
        if row is None:
            raise KeyError(dataset_id)

        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        labels = ["Mahalanobis", "Pareto k", "PSIS ESS"]
        values = [
            row.amortized.get(
                "mahalanobis_statistic",
                row.amortized.get("mahalanobis_distance", np.nan),
            ),
            row.psis.get("pareto_k", np.nan),
            row.psis.get("ess", np.nan),
        ]
        ax.bar(labels, values, color=["#64748b", "#d97706", "#0ea5e9"])
        ax.set_title(f"Dataset {dataset_id} diagnostics ({row.status})")
        ax.axhline(0.7, linestyle="--", linewidth=1, color="gray")
        fig.tight_layout()
        return fig

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.status] = counts.get(r.status, 0) + 1
        return counts

    def with_replacements(
        self, updates: Mapping[int, DatasetResult]
    ) -> "WorkflowReport":
        replaced = tuple(updates.get(r.dataset_id, r) for r in self.results)
        return replace(self, results=replaced)
