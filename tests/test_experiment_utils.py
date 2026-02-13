import tempfile
import time
from pathlib import Path

import pytest
from sbi_mcmc.utils.experiment_utils import PickleStatLogger


@pytest.fixture
def logger_with_tempfile():
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = Path(temp_dir) / "test_stats.pkl"
        logger = PickleStatLogger(filepath)
        yield logger, filepath


def test_initialization_and_loading(logger_with_tempfile):
    logger, _ = logger_with_tempfile
    assert logger.data == {}  # Should initialize with empty data


def test_save_and_update(logger_with_tempfile):
    logger, filepath = logger_with_tempfile

    # Update data
    logger.update("run_001", {"accuracy": 0.9})
    assert logger.data == {"run_001": {"accuracy": 0.9}}

    # Ensure data is saved to file
    assert filepath.exists()

    # Reload logger and check data persistence
    new_logger = PickleStatLogger(filepath)
    assert new_logger.data == {"run_001": {"accuracy": 0.9}}


def test_get_method(logger_with_tempfile):
    logger, _ = logger_with_tempfile

    # Update data
    logger.update("run_001", {"accuracy": 0.9})
    assert logger.get("run_001") == {"accuracy": 0.9}
    assert logger.get("non_existent_key") is None


def test_timer_context_manager(logger_with_tempfile):
    logger, _ = logger_with_tempfile

    # Use timer context manager
    with logger.timer("run_001"):
        time.sleep(0.1)  # Simulate some work

    # Check if wall-clock time is logged
    assert "wall_time" in logger.data
    assert "run_001" in logger.data["wall_time"]
    assert logger.data["wall_time"]["run_001"] > 0
    assert (
        abs(logger.data["wall_time"]["run_001"] - 0.1) < 0.02
    )  # Allow 0.1s tolerance


def test_timer_exception_handling(logger_with_tempfile):
    logger, _ = logger_with_tempfile

    # Use timer context manager
    try:
        with logger.timer("run_001"):
            raise ValueError("Test exception")
    except ValueError:
        pass

    assert "wall_time" not in logger.data
