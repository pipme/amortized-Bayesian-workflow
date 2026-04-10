from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from .base import SamplerBackend, SamplerRequest, SamplerResult
from .resolve import get_backend

if TYPE_CHECKING:
    from .blackjax_chees_hmc import BlackJAXCheesHMCBackend
    from .blackjax_nuts import BlackJAXNUTSBackend
    from .tfp_chees_hmc import TFPCheesHMCBackend

__all__ = [
    "BlackJAXCheesHMCBackend",
    "BlackJAXNUTSBackend",
    "SamplerBackend",
    "SamplerRequest",
    "SamplerResult",
    "TFPCheesHMCBackend",
    "get_backend",
]


def __getattr__(name: str):
    if name == "BlackJAXCheesHMCBackend":
        mod = import_module(".blackjax_chees_hmc", __name__)
        return getattr(mod, name)
    if name == "BlackJAXNUTSBackend":
        mod = import_module(".blackjax_nuts", __name__)
        return getattr(mod, name)
    if name == "TFPCheesHMCBackend":
        mod = import_module(".tfp_chees_hmc", __name__)
        return getattr(mod, name)
    raise AttributeError(name)
