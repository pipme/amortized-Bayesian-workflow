from pathlib import Path

from amortized_bayesian_workflow.config import ArtifactLayout


def test_artifact_layout_paths(tmp_path: Path):
    layout = ArtifactLayout(root=tmp_path, task_name="GEV", run_name="demo")
    layout.ensure()

    assert layout.datasets_dir.exists()
    assert layout.models_dir.exists()
    assert layout.diagnostics_dir.exists()
    assert layout.mcmc_dir.exists()

