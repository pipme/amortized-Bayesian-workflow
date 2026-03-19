from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .bayesflow import BayesFlowAmortizedPosterior


@dataclass(frozen=True)
class BayesFlowPresetConfig:
    """Thin convenience config for common BayesFlow defaults.

    This is intentionally minimal and does not replace BayesFlow's API.
    Users can always instantiate BayesFlow objects directly and wrap them with
    `BayesFlowAmortizedPosterior`.
    """

    summary_kind: str = "set"  # "set" or "none"
    summary_dim: int = 32
    coupling_depth: int = 4
    subnet_widths: tuple[int, ...] = (128, 128)
    optimizer: str | None = "adam"
    compile_kwargs: Mapping[str, Any] = field(default_factory=dict)


def build_default_bayesflow_continuous_approximator(
    *,
    preset: BayesFlowPresetConfig | None = None,
):
    """Build a BayesFlow `ContinuousApproximator` with simple defaults.

    This helper exists only to reduce boilerplate in examples and notebooks.
    For advanced use, instantiate BayesFlow objects directly.
    """

    import bayesflow as bf

    cfg = preset or BayesFlowPresetConfig()
    adapter = bf.ContinuousApproximator.build_adapter(
        inference_variables=["parameters"],
        summary_variables=["observables"],
    )
    inference_network = bf.networks.CouplingFlow(
        subnet="mlp",
        depth=cfg.coupling_depth,
        subnet_kwargs={"widths": tuple(cfg.subnet_widths)},
    )

    if cfg.summary_kind == "none":
        summary_network = None
    elif cfg.summary_kind == "set":
        summary_network = bf.networks.DeepSet(summary_dim=cfg.summary_dim)
    else:
        raise ValueError(f"Unsupported summary_kind: {cfg.summary_kind}")

    approximator = bf.ContinuousApproximator(
        adapter=adapter,
        inference_network=inference_network,
        summary_network=summary_network,
    )
    if cfg.optimizer is not None:
        approximator.compile(optimizer=cfg.optimizer, **dict(cfg.compile_kwargs))
    return approximator


def make_default_bayesflow_posterior(
    *,
    preset: BayesFlowPresetConfig | None = None,
) -> BayesFlowAmortizedPosterior:
    """Build and wrap a BayesFlow approximator with small, explicit defaults."""

    approximator = build_default_bayesflow_continuous_approximator(preset=preset)
    return BayesFlowAmortizedPosterior(approximator)
