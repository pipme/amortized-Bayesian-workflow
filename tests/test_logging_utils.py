from __future__ import annotations

import logging

import pytest

from amortized_bayesian_workflow.logging_utils import configure_logging


def test_configure_logging_sets_package_logger_level():
    logger = logging.getLogger("amortized_bayesian_workflow")
    old = logger.level
    try:
        configure_logging("ERROR")
        assert logger.level == logging.ERROR
    finally:
        logger.setLevel(old)


def test_configure_logging_optionally_sets_external_logger_level():
    pymc_logger = logging.getLogger("pymc.sampling.forward")
    old = pymc_logger.level
    try:
        configure_logging("WARNING", include_external_loggers=False)
        assert pymc_logger.level == old

        configure_logging("ERROR", include_external_loggers=True)
        assert pymc_logger.level == logging.ERROR
    finally:
        pymc_logger.setLevel(old)


def test_configure_logging_rejects_invalid_level():
    with pytest.raises(ValueError, match="log level"):
        configure_logging("NOT_A_LEVEL")
