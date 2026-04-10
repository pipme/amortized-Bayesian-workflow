from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from bayesflow import ContinuousApproximator

from .base import AmortizedDraws


@dataclass
class BayesFlowAmortizedPosterior:
    """Thin wrapper between a trained BayesFlow ContinuousApproximator and the
    ABW inference runner.

    Delegates to BayesFlow's high-level `sample`, `log_prob`, and `summarize`
    methods, which correctly handle the full adapter + summary network pipeline
    and the change-of-variables term from standardization. Samples and log
    probabilities are returned in the parameter space
    which is consistent with what the task's log posterior expects.
    """

    approximator: ContinuousApproximator
    observable_key: str = "observables"
    training_summaries: np.ndarray | None = field(default=None, repr=False)

    _APPROXIMATOR_FILENAME: ClassVar[str] = "approximator.keras"
    _METADATA_FILENAME: ClassVar[str] = "metadata.json"
    _TRAINING_SUMMARIES_FILENAME: ClassVar[str] = "training_summaries.npy"

    def store_training_summaries(self, observables: np.ndarray) -> None:
        """Compute and cache learned summary features for Mahalanobis OOD calibration.

        Call once after training so WorkflowRunner can use pre-computed summaries
        instead of re-simulating reference data.
        """
        self.training_summaries = self.summary_statistics(
            np.asarray(observables)
        )

    def summary_statistics(self, observations: np.ndarray) -> np.ndarray:
        """Combined inference conditions used for Mahalanobis OOD diagnostics.

        Produces the full feature vector that feeds the inference network,
        meaning the OOD check operates in the same feature space as the inference network.
        """
        import keras

        obs = np.asarray(observations)
        if obs.ndim == 1:
            obs = obs[None, ...]

        resolved_conditions, adapted, summary_outputs = (
            self.approximator._prepare_conditions({self.observable_key: obs})
        )
        # summary_variables = data.get("summary_variables")
        # inference_conditions = data.get("inference_conditions")

        # if self.approximator.summary_network is not None:
        #     if summary_variables is None:
        #         raise ValueError(
        #             "BayesFlow adapter did not produce summary_variables."
        #         )
        #     summary_outputs = self.approximator.summary_network(
        #         summary_variables
        #     )
        #     if inference_conditions is None:
        #         inference_conditions = summary_outputs
        #     else:
        #         inference_conditions = keras.ops.concatenate(
        #             [inference_conditions, summary_outputs], axis=-1
        #         )

        # if inference_conditions is None:
        #     return obs.reshape(obs.shape[0], -1)

        return keras.ops.convert_to_numpy(resolved_conditions)

    def sample_and_log_prob(
        self,
        observation: np.ndarray,
        *,
        num_samples: int,
        seed: int,
    ) -> AmortizedDraws:
        obs = np.asarray(observation)

        # approximator.sample handles adapter + summary network + inverse
        # standardization, returning samples in the original parameter space.
        # Shape per key: (1, num_samples, param_dim) for one condition.
        samples_dict = self.approximator.sample(
            num_samples=num_samples,
            conditions={self.observable_key: obs[None, ...]},
            seed=seed,
        )
        # Concatenate all parameter groups (axis=-1) in deterministic key order.
        samples = np.concatenate(
            [v[0] for v in samples_dict.values()], axis=-1
        )

        # approximator.log_prob handles adapter + summary network + the
        # change-of-variables correction for standardization, returning log q
        # in the original parameter space — matching log_p from the task.
        obs_batch = np.repeat(obs[None, ...], num_samples, axis=0)
        log_prob = self.approximator.log_prob(
            {
                self.observable_key: obs_batch,
                **{k: v[0] for k, v in samples_dict.items()},
            }
        )

        return AmortizedDraws(
            samples=samples,
            log_prob=np.asarray(log_prob).ravel(),
            metadata={"seed": seed},
        )

    def save(self, path: str | Path) -> None:
        """Save the approximator and wrapper metadata to a directory.

        The approximator is serialized with Keras and wrapper metadata is stored
        as JSON / NPY sidecar files.
        """
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)

        approximator_path = target / self._APPROXIMATOR_FILENAME
        self.approximator.save(str(approximator_path))

        metadata = {
            "observable_key": self.observable_key,
            "has_training_summaries": self.training_summaries is not None,
        }
        (target / self._METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        if self.training_summaries is not None:
            np.save(
                target / self._TRAINING_SUMMARIES_FILENAME,
                self.training_summaries,
            )

    @classmethod
    def load(cls, path: str | Path) -> BayesFlowAmortizedPosterior:
        """Load a saved BayesFlowAmortizedPosterior from a directory."""
        import keras

        source = Path(path)
        metadata_path = source / cls._METADATA_FILENAME
        approximator_path = source / cls._APPROXIMATOR_FILENAME

        if not approximator_path.exists():
            raise FileNotFoundError(
                f"Approximator file not found: {approximator_path}"
            )
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}"
            )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        approximator = keras.saving.load_model(str(approximator_path))

        obj = cls(
            approximator=approximator,
            observable_key=metadata.get("observable_key", "observables"),
        )

        training_summaries_path = source / cls._TRAINING_SUMMARIES_FILENAME
        if training_summaries_path.exists():
            obj.training_summaries = np.load(training_summaries_path)

        return obj
