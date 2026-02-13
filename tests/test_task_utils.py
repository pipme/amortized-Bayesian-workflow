import numpy as np
from sbi_mcmc.tasks import CustomDDM, PsychometricTask
from sbi_mcmc.tasks.tasks_utils import get_task_logp_func


def test_dynamic_logp():
    tasks = [CustomDDM(dt=0.0001), PsychometricTask(overdispersion=True)]

    for task in tasks:
        lp_fn = get_task_logp_func(task)

        lp_fn_dynamic = get_task_logp_func(
            task,
            static=False,
            pymc_model=task.setup_pymc_model(),
        )
        if isinstance(task, CustomDDM):
            params = np.array(
                [
                    [
                        0.59,
                        0.86,
                        -0.06,
                        0.02,
                        -1.11,
                        0.22,
                    ],
                    [
                        0.58,
                        0.84,
                        -0.01,
                        0.12,
                        -1.16,
                        0.01,
                    ],
                ],
            )
        elif isinstance(task, PsychometricTask):
            params = np.array(
                [
                    np.ones(task.D),
                    np.array([1.0, 0.5, 0.1, 0.2, 0.3])[
                        : task.D
                    ],  # different param values such that order matters
                    np.zeros(task.D),
                ],
            )
        # Call the static logp function
        lp_fn_static_values = lp_fn(params)
        # Call the dynamic logp function
        lp_fn_dynamic_values = lp_fn_dynamic(
            params, task.observation_to_pymc_data()
        )
        # Check if the values are close
        assert np.allclose(lp_fn_static_values, lp_fn_dynamic_values)
