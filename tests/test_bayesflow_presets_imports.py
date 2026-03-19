from amortized_bayesian_workflow.approximators import BayesFlowPresetConfig


def test_bayesflow_preset_config_defaults():
    cfg = BayesFlowPresetConfig()
    assert cfg.summary_kind in {"set", "none"}
    assert cfg.optimizer == "adam"

