from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Iterator, TypeVar

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
        yield from ex.map(fn, items)


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
