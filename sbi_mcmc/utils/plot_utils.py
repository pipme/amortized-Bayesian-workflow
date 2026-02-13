import matplotlib.pyplot as plt
import numpy as np
from corner import corner, overplot_lines, overplot_points


def corner_plot(
    gt_samples,
    algo_samples,
    title=None,
    txt=None,
    save_as=None,
    labels=None,
    transform=None,
    point=None,
    **kwargs,
):
    if transform is not None:
        gt_samples = transform(gt_samples)
        algo_samples = transform(algo_samples)
    D = gt_samples.shape[1]
    fig = corner(
        gt_samples,
        color="tab:orange",
        hist_kwargs={"density": True},
    )
    var_names = kwargs.get(
        "var_names", [f"$\\theta_{{ {i} }}$" for i in range(D)]
    )
    corner(
        algo_samples,
        fig=fig,
        color="tab:blue",
        contour_kwargs={"linestyles": "dashed"},
        hist_kwargs={"density": True},
        labels=var_names,
    )
    for ax in fig.get_axes():
        ax.tick_params(axis="both", labelsize=12)
    if labels is None:
        labels = ["Ground truth", "Algo"]

    if point is not None:
        if transform is not None:
            point = transform(point).squeeze()

        overplot_points(fig, point[None], marker="s", color="r")
        overplot_lines(fig, point, color="r")
    lgd = fig.legend(
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.8, 0.6),
    )
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    if txt is not None:
        text_art = fig.text(
            0.0, -0.10, txt, wrap=True, horizontalalignment="left", fontsize=12
        )
        extra_artists = (text_art, lgd)
    else:
        extra_artists = (lgd,)
    if save_as is not None:
        fig.savefig(
            save_as,
            dpi=kwargs.get("dpi", 300),
            bbox_extra_artists=extra_artists,
            bbox_inches="tight",
        )
    return fig


def plot_true_vs_est(true, estimated, param_names=None):
    num_datasets = true.shape[0]
    num_params = true.shape[-1]

    fig, axes = plt.subplots(1, num_params, figsize=(4 * num_params, 4))

    for i in range(num_params):
        lower_est = np.percentile(estimated[:, :, i], 2.5, axis=1)
        upper_est = np.percentile(estimated[:, :, i], 97.5, axis=1)
        median_est = np.median(estimated[:, :, i], axis=1)

        # line from lower_est to upper_est
        for j in range(num_datasets):
            axes[i].plot(
                [true[j, i], true[j, i]],
                [lower_est[j], upper_est[j]],
                "k-",
                alpha=0.2,
            )
        axes[i].plot(true[:, i], median_est, "o")
        axes[i].plot(true[:, i], true[:, i], "k--")
        axes[i].set_xlabel("True", fontsize="x-large")
        axes[i].set_ylabel("Estimated", fontsize="x-large")

        axes[i].set_title(
            f"Parameter {i + 1}" if param_names is None else param_names[i],
            fontsize="x-large",
        )

    plt.tight_layout()
    plt.show()
