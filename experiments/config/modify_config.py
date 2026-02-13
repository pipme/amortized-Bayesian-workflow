"""This script generates the config.yaml file using Hydra and OmegaConf. We used it for parsing command line arguments."""

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.1", config_path=".", config_name="default_config")
def main(cfg: DictConfig):
    # Save updated config to a YAML file
    save_path = Path(__file__).resolve().parent / "config.yaml"
    with open(save_path, "w") as f:  # noqa: PTH123
        OmegaConf.save(config=cfg, f=f.name)


if __name__ == "__main__":
    main()
