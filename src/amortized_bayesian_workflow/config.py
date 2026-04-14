from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ArtifactLayout:
    """Filesystem layout for workflow outputs."""

    root: Path
    task_name: str
    run_name: str = "default"

    @property
    def task_dir(self) -> Path:
        return self.root / self.task_name

    @property
    def run_dir(self) -> Path:
        return self.task_dir / self.run_name

    @property
    def datasets_dir(self) -> Path:
        return self.run_dir / "datasets"

    @property
    def models_dir(self) -> Path:
        return self.run_dir / "models"

    @property
    def diagnostics_dir(self) -> Path:
        return self.run_dir / "diagnostics"

    @property
    def mcmc_dir(self) -> Path:
        return self.run_dir / "mcmc"

    def ensure(self) -> "ArtifactLayout":
        for path in (
            self.datasets_dir,
            self.models_dir,
            self.diagnostics_dir,
            self.mcmc_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class WorkflowConfig:
    """High-level workflow settings independent of a specific backend."""

    num_train_simulations: int = 10_000
    num_validation_simulations: int = 1_000
    num_diagnostic_simulations: int = 200
    num_amortized_draws: int = 2_000
    batch_size: int = 256
    epochs: int = 100

    mahalanobis_alpha: float = 0.05
    force_psis_for_all_datasets: bool = False
    store_psis_resampled_draws: bool = False

    force_mcmc: bool = False
    mcmc_backend: str = "nuts"
    mcmc_backend_options: dict[str, object] = field(default_factory=dict)

    persist_dataset_results: bool = False
    rewrite_persisted_dataset_results: bool = False
    parallel_workers: int | None = None
    parallel_mode: str = "none"
    seed: int = 0
