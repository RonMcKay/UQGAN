from sacred import Ingredient
import torch.nn as nn

from utils import load_config_from_checkpoint

from .fc import MetaClassifier

meta_models = Ingredient("meta_model")


@meta_models.config
def config():
    cfg = dict()
    cfg["name"] = "fc"
    cfg["checkpoint"] = None

    if cfg["checkpoint"] is not None:
        # try loading a saved configuration
        config_update = load_config_from_checkpoint(checkpoint_path=cfg["checkpoint"])
        config_update.pop("checkpoint", None)
        cfg.update(config_update)

        del config_update


@meta_models.capture
def load_meta_model(cfg, _log, **kwargs) -> nn.Module:
    if cfg["name"].lower() == "fc":
        args = dict()
        args.update(cfg)
        args.udpate(kwargs)

        meta_classifier = MetaClassifier(**args)
    else:
        _log.error("Unknown model '{}'".format(cfg["name"]))
        raise ValueError("Unknown model '{}'".format(cfg["name"]))

    return meta_classifier
