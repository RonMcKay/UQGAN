from copy import deepcopy

from sacred import Ingredient
import torch.nn as nn

from bnn_models.lenet import BLeNet
from bnn_models.resnet import BClassifier
from datasets import datasets
from utils import load_config_from_checkpoint

bnn_models = Ingredient("bnn_model", ingredients=(datasets,))


@bnn_models.config
def config():
    cfg = dict()
    cfg["name"] = "blenet"
    cfg["checkpoint"] = None

    if cfg["name"].lower() in ("bresnet",):
        cfg["n_featuremaps"] = 32
        cfg["norm"] = "layernorm"

    if cfg["checkpoint"] is not None:
        # try loading a saved configuration
        config_update = load_config_from_checkpoint(checkpoint_path=cfg["checkpoint"])
        config_update.pop("checkpoint", None)
        cfg.update(config_update)

        del config_update


@bnn_models.capture
def load_bnn_model(cfg, dataset, _log, **kwargs) -> nn.Module:
    cfg = deepcopy(cfg)
    cfg.update(kwargs)
    if cfg["name"].lower() == "bresnet":
        args = dict(
            cl_dim=dataset["cfg"]["cl_dim"],
            image_channels=dataset["cfg"]["image_channels"],
            image_size=dataset["cfg"]["image_size"],
        )
        args.update(cfg)

        classifier = BClassifier(**args)
    elif cfg["name"].lower() == "blenet":
        args = dict(
            cl_dim=dataset["cfg"]["cl_dim"],
            image_channels=dataset["cfg"]["image_channels"],
            image_size=dataset["cfg"]["image_size"],
        )
        args.update(cfg)

        classifier = BLeNet(**args)
    else:
        _log.error("Unknown model '{}'".format(cfg["name"]))
        raise ValueError("Unknown model '{}'".format(cfg["name"]))

    return classifier
