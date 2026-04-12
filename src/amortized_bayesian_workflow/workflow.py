from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .approximators.base import AmortizedPosterior
from .backends import SamplerRequest, get_backend
from .backends.resolve import BackendFactory
from .config import ArtifactLayout, WorkflowConfig
from .diagnostics import MahalanobisOODResult, MahalanobisReference
from .psis import compute_psis, resample_with_weights
from .report import DatasetResult, WorkflowReport
from .tasks.base import WorkflowTask
from .utils import map_parallel, read_from_file, save_to_file


@dataclass
class WorkflowRunner:
    """User-facing end-to-end amortized Bayesian workflow.

    The runner supports:
    - simulator-based training via `fit()`
    - dataset-level PSIS diagnostics (parallelized)
    - optional MCMC refinement for failed/flagged datasets
    - notebook-friendly result collection and retry workflows
    """

    task: WorkflowTask
    approximator: AmortizedPosterior
    config: WorkflowConfig
    layout: ArtifactLayout | None = None
    backend_factories: dict[str, BackendFactory] | None = None
    diagnostic_summary_fn: Callable[[np.ndarray], np.ndarray] | None = None

    def __post_init__(self) -> None:
        if self.backend_factories is None:
            self.backend_factories = {}
        self._mahalanobis_reference: MahalanobisReference | None = None
        self._mahalanobis_ood_by_dataset: dict[int, MahalanobisOODResult] = {}

    def prepare(self) -> None:
        if self.layout is not None:
            self.layout.ensure()

    def run(self, observations: Sequence[np.ndarray]) -> WorkflowReport:
        self.prepare()
        indexed = [(i, np.asarray(obs)) for i, obs in enumerate(observations)]
        results_by_id: dict[int, DatasetResult] = {}

        if self.config.persist_dataset_results:
            results_by_id.update(self._load_saved_dataset_results())

        self._mahalanobis_ood_by_dataset = {}
        if not self.config.force_psis_for_all_datasets and len(indexed) > 0:
            self._calibrate_mahalanobis_reference()
            stacked_observations = np.asarray([obs for _, obs in indexed])
            all_summaries = self._summary_statistics(stacked_observations)
            if self._mahalanobis_reference is None:
                raise RuntimeError(
                    "Mahalanobis reference was not initialized after calibration."
                )
            ood_results = self._mahalanobis_reference.evaluate_batch(
                all_summaries,
                alpha=self.config.mahalanobis_alpha,
            )
            self._mahalanobis_ood_by_dataset = {
                dataset_id: ood
                for (dataset_id, _), ood in zip(indexed, ood_results)
            }

        indexed_with_ood = [
            (
                dataset_id,
                observation,
                self._mahalanobis_ood_by_dataset.get(dataset_id),
            )
            for dataset_id, observation in indexed
            if dataset_id not in results_by_id
        ]
        parallel_mode = (
            "thread"
            if self.config.parallel_mode == "auto"
            else self.config.parallel_mode
        )
        for result in map_parallel(
            self._process_one_dataset,
            indexed_with_ood,
            mode=parallel_mode,
            max_workers=self.config.parallel_workers,
        ):
            results_by_id[result.dataset_id] = result
            if self.config.persist_dataset_results:
                self._save_dataset_result(result)

        results = tuple(results_by_id[dataset_id] for dataset_id, _ in indexed)
        return WorkflowReport(
            results=results,
            config=asdict(self.config),
            metadata={"task_name": self.task.metadata.name},
        )

    def _summary_statistics(self, observations: np.ndarray) -> np.ndarray:
        obs = np.asarray(observations)
        summary_method = getattr(self.approximator, "summary_statistics", None)
        if callable(summary_method):
            return np.asarray(summary_method(obs), dtype=float)
        else:
            raise ValueError(
                "Amortized approximator does not provide a summary_statistics method for diagnostics."
            )

    def _calibrate_mahalanobis_reference(
        self,
    ) -> MahalanobisReference | None:
        precomputed = getattr(self.approximator, "training_summaries", None)
        if precomputed is not None and np.asarray(precomputed).shape[0] >= 2:
            self._mahalanobis_reference = (
                MahalanobisReference.from_training_summaries(
                    np.asarray(precomputed, dtype=float)
                )
            )
        else:
            raise ValueError(
                "Amortized approximator does not provide summary statistics for training datasets, cannot calibrate Mahalanobis reference for diagnostics."
            )

    def retry_failed(
        self,
        report: WorkflowReport,
        observations: Sequence[np.ndarray],
        *,
        dataset_ids: Iterable[int] | None = None,
        config_override: dict[str, Any] | None = None,
    ) -> WorkflowReport:
        ids = (
            set(dataset_ids)
            if dataset_ids is not None
            else {r.dataset_id for r in report.failed_datasets()}
        )
        if not ids:
            return report

        original_config = self.config
        if config_override:
            self.config = WorkflowConfig(
                **(asdict(self.config) | dict(config_override))
            )
        try:
            rerun_results = {}
            for dataset_id in sorted(ids):
                precomputed_ood = self._mahalanobis_ood_by_dataset.get(
                    dataset_id
                )
                rerun_results[dataset_id] = self._process_one_dataset(
                    (
                        dataset_id,
                        np.asarray(observations[dataset_id]),
                        precomputed_ood,
                    )
                )
            return report.with_replacements(rerun_results)
        finally:
            self.config = original_config

    def _process_one_dataset(
        self,
        item: tuple[int, np.ndarray, MahalanobisOODResult | None],
    ) -> DatasetResult:
        dataset_id, observation, precomputed_ood = item
        base_seed = self.config.seed + dataset_id * 1009
        try:
            amortized = self.approximator.sample_and_log_prob(
                observation,
                num_samples=self.config.num_amortized_draws,
                seed=base_seed,
            )
            q_samples = np.asarray(amortized.samples)
            log_q = np.asarray(amortized.log_prob)
            if q_samples.ndim != 2:
                raise ValueError(
                    "Amortized samples must have shape (num_samples, parameter_dim)."
                )
            if log_q.shape[0] != q_samples.shape[0]:
                raise ValueError("Amortized log_prob length mismatch.")

            log_post_vec = self.task.vectorized_log_posterior_fn(observation)
            log_post_single = self.task.single_log_posterior_fn(observation)
            log_target = np.asarray(log_post_vec(q_samples), dtype=float)
            if log_target.shape[0] != q_samples.shape[0]:
                raise ValueError(
                    "Vectorized log posterior returned wrong shape."
                )

            amortized_payload: dict[str, Any] = {
                "num_samples": int(q_samples.shape[0]),
                "draws": q_samples,
            }
            send_to_psis = bool(self.config.force_psis_for_all_datasets)
            if self.config.force_psis_for_all_datasets:
                amortized_payload.update(
                    {
                        "ood_rejected": True,
                        "ood_rejection_reason": "force_psis_for_all_datasets",
                        "mahalanobis_alpha": float(
                            self.config.mahalanobis_alpha
                        ),
                    }
                )
            elif precomputed_ood is not None:
                amortized_payload.update(
                    {
                        "mahalanobis_statistic": precomputed_ood.statistic,
                        "mahalanobis_cutoff": precomputed_ood.cutoff,
                        "mahalanobis_alpha": precomputed_ood.alpha,
                        "ood_rejected": precomputed_ood.rejected,
                    }
                )
                send_to_psis = bool(precomputed_ood.rejected)
            else:
                amortized_payload.update(
                    {
                        "ood_rejected": True,
                        "ood_rejection_reason": "mahalanobis_unavailable",
                        "mahalanobis_alpha": float(
                            self.config.mahalanobis_alpha
                        ),
                    }
                )
                send_to_psis = True

            if self.config.force_mcmc:
                send_to_psis = True

            psis_payload: dict[str, Any] = {}
            mcmc_payload: dict[str, Any] = {}

            if not send_to_psis and not self.config.force_mcmc:
                status = "success"
                message = "Mahalanobis diagnostic accepted amortized posterior draws."
                return DatasetResult(
                    dataset_id=dataset_id,
                    status=status,
                    message=message,
                    posterior_source="amortized",
                    amortized=amortized_payload,
                    psis=psis_payload,
                    mcmc=mcmc_payload,
                )

            psis = compute_psis(log_target=log_target, log_proposal=log_q)
            psis_payload = asdict(psis)
            psis_payload["posterior_num_draws"] = int(
                self.config.num_amortized_draws
            )
            psis_payload["posterior_seed"] = int(base_seed + 1)

            if self.config.store_psis_resampled_draws:
                psis_payload["resampled_draws"] = resample_with_weights(
                    q_samples,
                    psis.smoothed_normalized_weights,
                    num_draws=self.config.num_amortized_draws,
                    seed=base_seed + 1,
                    replace=True,
                )

            needs_mcmc = self.config.force_mcmc or psis.needs_mcmc()
            if needs_mcmc:
                backend_name = self._resolve_mcmc_backend_name()
                backend = get_backend(
                    backend_name,
                    extra_factories=self.backend_factories,
                )
                top_k = max(
                    2, min(self.config.mcmc_init_top_k, q_samples.shape[0])
                )
                init_idx = np.argsort(-psis.smoothed_log_weights)[:top_k]
                backend_options = self._prepare_mcmc_options(
                    backend_name=backend_name,
                    default_num_superchains=top_k,
                )
                mcmc_result = backend.run(
                    SamplerRequest(
                        initial_positions=q_samples[init_idx],
                        log_prob_fn=log_post_vec,
                        single_log_prob_fn=log_post_single,
                        num_warmup=self.config.mcmc_warmup,
                        num_samples=self.config.mcmc_num_samples,
                        seed=base_seed + 2,
                        options=backend_options,
                    )
                )
                mcmc_draws = np.asarray(mcmc_result.draws).reshape(
                    -1, mcmc_result.draws.shape[-1]
                )
                mcmc_payload = {
                    "backend": mcmc_result.backend,
                    "draws": mcmc_draws,
                    **dict(mcmc_result.diagnostics),
                    **dict(mcmc_result.metadata),
                }
                preferred_metric = (
                    "nested_rhat" if "chees" in backend_name else "rhat"
                )
                convergence_metric = preferred_metric
                rhat_max = mcmc_result.convergence_value(
                    metric=convergence_metric
                )
                mcmc_payload["convergence_metric"] = convergence_metric
                mcmc_payload["convergence_value"] = rhat_max
                status = (
                    "success"
                    if mcmc_result.is_converged(metric=convergence_metric)
                    else "needs_review"
                )
                message = (
                    "MCMC refinement completed."
                    if status == "success"
                    else "MCMC completed but diagnostics suggest review."
                )
                posterior_source = "mcmc"
            else:
                status = "success"
                message = "PSIS accepted amortized draws; use PSIS weights for expectation estimates."
                posterior_source = "psis_weighted"

            return DatasetResult(
                dataset_id=dataset_id,
                status=status,
                message=message,
                posterior_source=posterior_source,
                amortized=amortized_payload,
                psis=psis_payload,
                mcmc=mcmc_payload,
            )
        except Exception as exc:
            return DatasetResult(
                dataset_id=dataset_id,
                status="failed",
                message="Workflow failed for dataset.",
                error=f"{type(exc).__name__}: {exc}",
            )

    def _resolve_mcmc_backend_name(self) -> str:
        sampler_name = self.config.mcmc_sampler.strip()
        if not sampler_name and self.config.mcmc_backend is not None:
            sampler_name = self.config.mcmc_backend
        if not sampler_name:
            sampler_name = "nuts"

        normalized = sampler_name.lower()
        aliases = {
            "nuts": "blackjax_nuts",
            "blackjax_nuts": "blackjax_nuts",
            "chees_hmc": "blackjax_chees_hmc",
            "blackjax_chees_hmc": "blackjax_chees_hmc",
            "tfp_chees_hmc": "tfp_chees_hmc",
        }
        return aliases.get(normalized, normalized)

    def _prepare_mcmc_options(
        self,
        *,
        backend_name: str,
        default_num_superchains: int,
    ) -> dict[str, Any]:
        options = dict(self.config.mcmc_backend_options)
        num_superchains = int(
            options.pop("num_superchains", default_num_superchains)
        )

        if "chees" in backend_name:
            num_subchains = int(
                options.pop(
                    "num_subchains_per_superchain",
                    options.pop("subchains_per_superchain", 1),
                )
            )
            return {
                "num_superchains": num_superchains,
                "subchains_per_superchain": num_subchains,
                **options,
            }

        return {
            "num_superchains": num_superchains,
            **options,
        }

    def _dataset_result_path(self, dataset_id: int) -> Path | None:
        if self.layout is None:
            return None
        return self.layout.datasets_dir / f"dataset_{int(dataset_id)}.dill"

    def _save_dataset_result(self, result: DatasetResult) -> None:
        path = self._dataset_result_path(result.dataset_id)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        save_to_file(result, path, use_pickle=False)

    def _load_saved_dataset_results(self) -> dict[int, DatasetResult]:
        if self.layout is None:
            return {}
        datasets_dir = self.layout.datasets_dir
        if not datasets_dir.exists():
            return {}

        loaded: dict[int, DatasetResult] = {}
        for path in sorted(datasets_dir.glob("dataset_*.dill")):
            try:
                result = read_from_file(path, use_pickle=False)
            except Exception:
                continue
            if isinstance(result, DatasetResult):
                loaded[int(result.dataset_id)] = result
        return loaded
