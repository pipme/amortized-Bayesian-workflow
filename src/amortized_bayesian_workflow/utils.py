from __future__ import annotations

from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path
from typing import Callable, Iterable, Iterator, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from corner import corner, overplot_lines, overplot_points

T = TypeVar("T")
U = TypeVar("U")


def map_parallel(
    fn: Callable[[T], U],
    items: Iterable[T],
    *,
    mode: str = "thread",
    max_workers: int | None = None,
) -> Iterator[U]:
    if mode == "none":
        for item in items:
            yield fn(item)
        return

    executor_cls = (
        ThreadPoolExecutor if mode == "thread" else ProcessPoolExecutor
    )
    with executor_cls(max_workers=max_workers) as ex:
        futures = [ex.submit(fn, item) for item in items]
        for future in as_completed(futures):
            yield future.result()


def save_to_file(data, file_path, use_pickle=True):
    """Save data to a file."""
    if file_path is None:
        return

    with Path(file_path).open("wb") as f:
        if use_pickle:
            import pickle

            pickle.dump(data, f)
        else:
            import dill

            dill.dump(data, f)
    print(f"Data saved to {file_path}")


def read_from_file(file_path, use_pickle=True):
    with Path(file_path).open("rb") as f:
        if use_pickle:
            import pickle

            data = pickle.load(f)
        else:
            import dill

            data = dill.load(f)
    return data


def corner_plot(
    samples_1: np.ndarray,
    samples_2: np.ndarray,
    *,
    title: str | None = None,
    txt: str | None = None,
    save_as: str | Path | None = None,
    labels: list[str] | None = None,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
    point: np.ndarray | None = None,
    dpi: int = 300,
    **kwargs,
) -> plt.Figure:
    """Create a corner plot comparing two sets of samples.

    Args:
        samples_1: First set of samples, shape (n_samples, n_dims).
        samples_2: Second set of samples, shape (n_samples, n_dims).
        title: Plot title.
        txt: Additional text to display on plot.
        save_as: File path to save figure.
        labels: Legend labels. Defaults to ["Samples 1", "Samples 2"].
        transform: Optional function to transform samples before plotting.
        point: Optional point to overlay on plot.
        dpi: DPI for saved figure. Defaults to 300.
        **kwargs: Additional arguments passed to corner().

    Returns:
        Matplotlib figure object.
    """
    if transform is not None:
        samples_1 = transform(samples_1)
        samples_2 = transform(samples_2)

    n_dims = samples_1.shape[1]
    var_names = kwargs.pop(
        "var_names", [f"$\\theta_{{ {i} }}$" for i in range(n_dims)]
    )

    fig = corner(
        samples_1,
        color="tab:orange",
        hist_kwargs={"density": True},
        **kwargs,
    )
    corner(
        samples_2,
        fig=fig,
        color="tab:blue",
        contour_kwargs={"linestyles": "dashed"},
        hist_kwargs={"density": True},
        labels=var_names,
        **kwargs,
    )

    for ax in fig.get_axes():
        ax.tick_params(axis="both", labelsize=12)

    if labels is None:
        labels = ["Samples 1", "Samples 2"]

    if point is not None:
        point_plot = (
            transform(point).squeeze()
            if transform is not None
            else point.squeeze()
        )
        overplot_points(fig, point_plot[None], marker="s", color="r")
        overplot_lines(fig, point_plot, color="r")

    lgd = fig.legend(
        labels=labels, loc="upper center", bbox_to_anchor=(0.8, 0.6)
    )
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()

    extra_artists = [lgd]
    if txt is not None:
        text_art = fig.text(
            0.0,
            -0.10,
            txt,
            wrap=True,
            horizontalalignment="left",
            fontsize=12,
        )
        extra_artists.append(text_art)

    if save_as is not None:
        fig.savefig(
            save_as,
            dpi=dpi,
            bbox_extra_artists=extra_artists,
            bbox_inches="tight",
        )

    return fig
