from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .approximators.base import AmortizedPosterior
from .backends import SamplerRequest, get_backend
from .backends.resolve import BackendFactory
from .config import ArtifactLayout, WorkflowConfig
from .diagnostics import MahalanobisReference
from .psis import compute_psis, resample_with_weights
from .report import DatasetResult, WorkflowReport
from .tasks.base import WorkflowTask
from .utils import map_parallel


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

    def prepare(self) -> None:
        if self.layout is not None:
            self.layout.ensure()

    def run(self, observations: Sequence[np.ndarray]) -> WorkflowReport:
        self.prepare()
        self._calibrate_mahalanobis_reference()
        indexed = [(i, np.asarray(obs)) for i, obs in enumerate(observations)]
        parallel_mode = (
            "thread"
            if self.config.parallel_mode == "auto"
            else self.config.parallel_mode
        )
        results = tuple(
            map_parallel(
                self._process_one_dataset,
                indexed,
                mode=parallel_mode,
                max_workers=self.config.parallel_workers,
            )
        )
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
                rerun_results[dataset_id] = self._process_one_dataset(
                    (dataset_id, np.asarray(observations[dataset_id]))
                )
            return report.with_replacements(rerun_results)
        finally:
            self.config = original_config

    def _process_one_dataset(
        self,
        item: tuple[int, np.ndarray],
    ) -> DatasetResult:
        dataset_id, observation = item
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
            elif self._mahalanobis_reference is not None:
                obs_summary = self._summary_statistics(observation[None, :])[0]
                ood = self._mahalanobis_reference.evaluate(
                    obs_summary, alpha=self.config.mahalanobis_alpha
                )
                amortized_payload.update(
                    {
                        "mahalanobis_statistic": ood.statistic,
                        "mahalanobis_cutoff": ood.cutoff,
                        "mahalanobis_alpha": ood.alpha,
                        "ood_rejected": ood.rejected,
                    }
                )
                send_to_psis = bool(ood.rejected)
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
                posterior_draws = q_samples[: self.config.mcmc_num_samples]
                status = "success"
                message = "Mahalanobis diagnostic accepted amortized posterior draws."
                return DatasetResult(
                    dataset_id=dataset_id,
                    status=status,
                    message=message,
                    posterior_draws=posterior_draws,
                    amortized_draws=q_samples,
                    amortized=amortized_payload,
                    psis=psis_payload,
                    mcmc=mcmc_payload,
                )

            psis = compute_psis(log_target=log_target, log_proposal=log_q)
            posterior_draws = resample_with_weights(
                q_samples,
                psis.normalized_weights,
                num_draws=self.config.mcmc_num_samples,
                seed=base_seed + 1,
            )

            psis_payload = {
                "pareto_k": psis.pareto_k,
                "ess": psis.ess,
                "num_samples": int(q_samples.shape[0]),
            }

            needs_mcmc = self.config.force_mcmc or psis.needs_mcmc(
                self.config.psis_k_threshold
            )
            if needs_mcmc:
                backend = get_backend(
                    self.config.mcmc_backend,
                    extra_factories=self.backend_factories,
                )
                top_k = max(
                    2, min(self.config.mcmc_init_top_k, q_samples.shape[0])
                )
                init_idx = np.argsort(-psis.smoothed_log_weights)[:top_k]
                mcmc_result = backend.run(
                    SamplerRequest(
                        initial_positions=q_samples[init_idx],
                        log_prob_fn=log_post_vec,
                        single_log_prob_fn=log_post_single,
                        num_warmup=self.config.mcmc_warmup,
                        num_samples=self.config.mcmc_num_samples,
                        seed=base_seed + 2,
                        options={"num_superchains": top_k},
                    )
                )
                posterior_draws = np.asarray(mcmc_result.draws).reshape(
                    -1, mcmc_result.draws.shape[-1]
                )
                mcmc_payload = {
                    "backend": mcmc_result.backend,
                    **dict(mcmc_result.diagnostics),
                    **dict(mcmc_result.metadata),
                }
                rhat = mcmc_payload.get(
                    "rhat", mcmc_payload.get("nested_rhat")
                )
                if isinstance(rhat, np.ndarray):
                    rhat_max = float(np.nanmax(rhat))
                elif rhat is None:
                    rhat_max = np.inf
                else:
                    rhat_max = float(rhat)
                status = (
                    "success"
                    if np.isfinite(rhat_max) and rhat_max <= 1.1
                    else "needs_review"
                )
                message = (
                    "MCMC refinement completed."
                    if status == "success"
                    else "MCMC completed but diagnostics suggest review."
                )
            else:
                status = "success"
                message = "PSIS accepted amortized posterior draws."

            return DatasetResult(
                dataset_id=dataset_id,
                status=status,
                message=message,
                posterior_draws=posterior_draws,
                amortized_draws=q_samples,
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
