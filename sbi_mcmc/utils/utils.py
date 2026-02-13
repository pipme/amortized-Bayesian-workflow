import random
from pathlib import Path

import dill
import numpy as np


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)


def save_to_file(data, file_path, use_pickle=True):
    """Save data to a file."""
    if file_path is None:
        return

    with Path(file_path).open("wb") as f:
        if use_pickle:
            import pickle

            pickle.dump(data, f)
        else:
            dill.dump(data, f)
    print(f"Data saved to {file_path}")


def read_from_file(file_path, use_pickle=True):
    with Path(file_path).open("rb") as f:
        if use_pickle:
            import pickle

            data = pickle.load(f)
        else:
            data = dill.load(f)
    return data
