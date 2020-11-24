from typing import Dict

from sacred import Ingredient
import torch.nn as nn

from datasets import datasets
from utils import load_config_from_checkpoint

from .sensoyetal2020 import AutoEncoder

vae_models = Ingredient("vae_model", ingredients=(datasets,))


@vae_models.config
def config():
    cfg = dict()
    cfg["name"] = "sensoyetal2020"
    cfg["latent_dim"] = 100
    cfg["checkpoint"] = None

    if cfg["checkpoint"] is not None:
        config_update = load_config_from_checkpoint(checkpoint_path=cfg["checkpoint"])
        config_update.pop("checkpoint", None)
        cfg.update(config_update)

        del config_update


@vae_models.capture
def load_vae_model(cfg: Dict, dataset: Dict, _log, **kwargs) -> nn.Module:
    if cfg["name"].lower() == "sensoyetal2020":
        args = dict(
            image_channels=dataset["cfg"]["image_channels"],
            image_size=dataset["cfg"]["image_size"],
        )
        args.update(cfg)
        args.update(kwargs)

        vae = AutoEncoder(**args)
    else:
        _log.error("Unknown model '{}'".format(cfg["name"]))
        raise ValueError("Unknown model '{}'".format(cfg["name"]))

    return vae
