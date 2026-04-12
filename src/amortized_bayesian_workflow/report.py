from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .psis import resample_with_weights
from .utils import read_from_file


@dataclass(frozen=True)
class DatasetResult:
    dataset_id: int
    status: str
    message: str
    posterior_source: str | None = None
    amortized: Mapping[str, Any] = field(default_factory=dict)
    psis: Mapping[str, Any] = field(default_factory=dict)
    mcmc: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def posterior_draws(self) -> np.ndarray | None:
        if self.posterior_source == "mcmc":
            draws = self.mcmc.get("draws")
            return None if draws is None else np.asarray(draws)
        if self.posterior_source == "amortized":
            draws = self.amortized.get("draws")
            return None if draws is None else np.asarray(draws)
        if self.posterior_source == "psis_weighted":
            resampled = self.psis.get("resampled_draws")
            if resampled is not None:
                return np.asarray(resampled)
            amortized = self.amortized.get("draws")
            weights = self.psis.get("smoothed_normalized_weights")
            if amortized is None or weights is None:
                return None
            amortized = np.asarray(amortized)
            num_draws = int(amortized.shape[0])
            return resample_with_weights(
                amortized,
                np.asarray(weights, dtype=float),
                num_draws=num_draws,
                seed=self.psis.get("posterior_seed"),
            )
        return None

    @property
    def amortized_draws(self) -> np.ndarray | None:
        draws = self.amortized.get("draws")
        return None if draws is None else np.asarray(draws)

    @property
    def mcmc_draws(self) -> np.ndarray | None:
        draws = self.mcmc.get("draws")
        return None if draws is None else np.asarray(draws)

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
                    "posterior_source": r.posterior_source,
                    "mcmc_backend": r.mcmc.get("backend"),
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

    @classmethod
    def from_dataset_results_dir(
        cls,
        path: str | Path,
        *,
        config: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        use_pickle: bool = False,
        strict: bool = False,
    ) -> "WorkflowReport":
        """Build a report post-hoc from per-dataset result artifacts.

        Expected files are `dataset_<id>.dill` by default, or `.pkl` when
        `use_pickle=True`.
        """
        root = Path(path)
        ext = ".pkl" if use_pickle else ".dill"
        files = sorted(root.glob(f"dataset_*{ext}"))

        # Fallback to the other extension when needed.
        if not files:
            alt = ".dill" if ext == ".pkl" else ".pkl"
            files = sorted(root.glob(f"dataset_*{alt}"))

        results: list[DatasetResult] = []
        invalid_files: list[str] = []
        for p in files:
            try:
                item = read_from_file(p, use_pickle=use_pickle)
            except Exception:
                if strict:
                    raise
                invalid_files.append(str(p.name))
                continue

            if isinstance(item, DatasetResult):
                results.append(item)
            else:
                if strict:
                    raise TypeError(
                        f"Artifact {p} is not a DatasetResult instance."
                    )
                invalid_files.append(str(p.name))

        results.sort(key=lambda r: int(r.dataset_id))

        md = dict(metadata or {})
        md.setdefault("source_dir", str(root))
        if invalid_files:
            md["invalid_artifacts"] = invalid_files

        return cls(
            results=tuple(results),
            config=dict(config or {}),
            metadata=md,
        )
