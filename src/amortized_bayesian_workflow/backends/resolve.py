from __future__ import annotations

from importlib import import_module

from .base import SamplerBackend


def _default_backend_factories() -> dict[str, object]:
    return {
        "blackjax_chees_hmc": lambda: getattr(
            import_module(".blackjax_chees_hmc", __package__),
            "BlackJAXCheesHMCBackend",
        )(),
        "blackjax_nuts": lambda: getattr(
            import_module(".blackjax_nuts", __package__),
            "BlackJAXNUTSBackend",
        )(),
        "tfp_chees_hmc": lambda: getattr(
            import_module(".tfp_chees_hmc", __package__),
            "TFPCheesHMCBackend",
        )(),
    }


def get_backend(
    name: str,
) -> SamplerBackend:
    factories = _default_backend_factories()

    if name not in factories:
        available = ", ".join(sorted(factories)) or "<none>"
        raise KeyError(f"Unknown backend '{name}'. Available: {available}")

    try:
        return factories[name]()
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Backend '{name}' is unavailable because optional dependencies are missing."
        ) from exc
