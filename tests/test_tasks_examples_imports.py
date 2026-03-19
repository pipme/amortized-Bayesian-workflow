from amortized_bayesian_workflow.tasks.examples.gev import gev_observation_default


def test_gev_observation_default_shape():
    obs = gev_observation_default()
    assert obs.ndim == 1
    assert obs.shape[0] == 65

