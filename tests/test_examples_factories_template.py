from pathlib import Path
import importlib.util

import numpy as np


def test_save_example_observations_npz(tmp_path: Path):
    module_path = Path(__file__).resolve().parents[1] / "examples" / "factories.py"
    spec = importlib.util.spec_from_file_location("examples_factories_template", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    out = module.save_example_observations_npz(tmp_path / "obs.npz")
    assert out.exists()
    data = np.load(out, allow_pickle=True)
    assert "observations" in data
