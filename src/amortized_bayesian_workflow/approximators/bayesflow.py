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
    """

    approximator: ContinuousApproximator
    observable_key: str = "observables"
    training_summaries: np.ndarray | None = field(default=None, repr=False)

    _APPROXIMATOR_FILENAME: ClassVar[str] = "approximator.keras"
    _METADATA_FILENAME: ClassVar[str] = "metadata.json"
    _TRAINING_SUMMARIES_FILENAME: ClassVar[str] = "training_summaries.npy"

    def store_training_summaries(self, observables: np.ndarray) -> None:
        """Compute and cache learned summary statistics of the training data.

        Call once after training so InferenceRunner can use pre-computed summaries
        instead of re-simulating reference data.
        """
        self.training_summaries = self.summary_statistics(
            np.asarray(observables)
        )

    def summary_statistics(self, observations: np.ndarray) -> np.ndarray:
        """
        Produces the full feature vector that feeds the inference network,
        meaning the OOD check in Inference phase (Step 1) operates in the same feature space as the inference network.
        """
        import keras

        obs = np.asarray(observations)
        if obs.ndim == 1:
            obs = obs[None, ...]

        resolved_conditions, adapted, summary_outputs = (
            self.approximator._prepare_conditions({self.observable_key: obs})
        )
        return keras.ops.convert_to_numpy(resolved_conditions)

    def sample_and_log_prob(
        self,
        observation: np.ndarray,
        *,
        num_samples: int,
        seed: int,
    ) -> AmortizedDraws:
        draws = self.sample_and_log_prob_batch(
            np.asarray(observation)[None, ...],
            num_samples=num_samples,
            seed=seed,
        )
        return draws[0]

    def sample_and_log_prob_batch(
        self,
        observations: np.ndarray,
        *,
        num_samples: int,
        seed: int,
    ) -> list[AmortizedDraws]:
        obs = np.asarray(observations)
        if obs.ndim == 1:
            obs = obs[None, ...]

        num_datasets = obs.shape[0]
        samples_dict = self.approximator.sample(
            num_samples=num_samples,
            conditions={self.observable_key: obs},
            seed=seed,
        )

        batched_samples = {k: np.asarray(v) for k, v in samples_dict.items()}
        samples = np.concatenate(list(batched_samples.values()), axis=-1)

        obs_batch = np.repeat(obs[:, None, ...], num_samples, axis=1)
        obs_batch = obs_batch.reshape((-1, *obs.shape[1:]))
        flat_samples = {
            k: v.reshape((-1, *v.shape[2:]))
            for k, v in batched_samples.items()
        }
        log_prob = self.approximator.log_prob(
            {
                self.observable_key: obs_batch,
                **flat_samples,
            }
        )
        log_prob = np.asarray(log_prob).reshape(num_datasets, num_samples)

        return [
            AmortizedDraws(
                samples=samples[i],
                log_prob=log_prob[i],
                metadata={"seed": seed, "dataset_index": i},
            )
            for i in range(num_datasets)
        ]

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
