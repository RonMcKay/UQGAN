from copy import deepcopy
import re

from sacred import Ingredient

from bnn_models.lenet import BLeNet
from bnn_models.resnet import BClassifier
from cls_models.lenet import LeNet
from cls_models.medium import MediumClassifier
from cls_models.resnet import Classifier
from cls_models.small import SmallClassifier
from cls_models.toy import ToyClassifier
from datasets import datasets
from utils import load_config_from_checkpoint

from .base import BaseClassifier
from .cls_utils import set_model_to_mode  # noqa: F401

cls_models = Ingredient("cls_model", ingredients=(datasets,))

CLS_MODELS = ("resnet", "bresnet", "lenet", "blenet", "medium", "small", "toy")


@cls_models.config
def config():
    cfg = dict()
    cfg["name"] = "lenet"
    cfg["checkpoint"] = None
    cfg["method"] = "softmax"
    cfg["mc_dropout"] = 0

    if cfg["name"].lower() in ("resnet",):
        cfg["n_featuremaps"] = 32
        cfg["norm"] = "layernorm"
    elif cfg["name"].lower() in ("blenet", "bresnet"):
        cfg["method"] = "bayes"

    if cfg.get("mc_dropout", 0) > 0:
        cfg["method"] = "mc-dropout"

    if cfg["checkpoint"] is not None:
        # try loading a saved configuration
        config_update = load_config_from_checkpoint(cfg["checkpoint"])
        config_update.pop("checkpoint", None)
        cfg.update(config_update)

        del config_update


@cls_models.capture
def load_cls_model(cfg, dataset, _log, cp_overwrite=None, **kwargs) -> BaseClassifier:
    if cp_overwrite is None:
        cp_overwrite = dict()
    cfg = deepcopy(cfg)

    if "checkpoint" in kwargs and cfg.get("checkpoint", None) is None:
        cp_config_update = load_config_from_checkpoint(kwargs["checkpoint"])
        cfg.update(cp_config_update)

    cfg.update(kwargs)

    if cfg["name"].lower() == "resnet":
        args = dict(
            cl_dim=dataset["cfg"]["cl_dim"],
            image_channels=dataset["cfg"]["image_channels"],
            image_size=dataset["cfg"]["image_size"],
        )
        args.update(cfg)

        if args.get("checkpoint", None) is not None:
            classifier = Classifier.load_from_checkpoint(
                args.get("checkpoint"), map_location="cpu", **cp_overwrite
            )
        else:
            classifier = Classifier(**args)
    elif cfg["name"].lower() == "bresnet":
        args = dict(
            cl_dim=dataset["cfg"]["cl_dim"],
            image_channels=dataset["cfg"]["image_channels"],
            image_size=dataset["cfg"]["image_size"],
        )
        args.update(cfg)

        if args.get("checkpoint", None) is not None:
            classifier = BClassifier.load_from_checkpoint(
                args.get("checkpoint"), map_location="cpu", **cp_overwrite
            )
        else:
            classifier = BClassifier(**args)
    elif re.match(r"resnet\d+", cfg["name"].lower()):
        _log.warning(
            "The 'resnet' models have been configured for an image size of 32x32. "
            "If you want to change this you have to adjust the 'blocks'"
        )
        try:
            args = dict(
                cl_dim=dataset["cfg"]["cl_dim"],
                image_channels=dataset["cfg"]["image_channels"],
                image_size=dataset["cfg"]["image_size"],
                n_featuremaps=64,
                blocks={"18": 2, "34": [4, 5, 7]}.get(
                    re.match(r"resnet(\d+)", cfg["name"].lower()).groups()[0]
                ),
            )
        except KeyError:
            _log.error(f"ResNet model '{cfg['name']}' not configured.")
            raise ValueError(f"ResNet model '{cfg['name']}' not configured.")
        args.update(cfg)

        if args.get("checkpoint", None) is not None:
            classifier = Classifier.load_from_checkpoint(
                args.get("checkpoint"), map_location="cpu", **cp_overwrite
            )
        else:
            classifier = Classifier(**args)
    elif cfg["name"].lower() == "lenet":
        args = dict(
            cl_dim=dataset["cfg"]["cl_dim"],
            image_channels=dataset["cfg"]["image_channels"],
            image_size=dataset["cfg"]["image_size"],
        )
        args.update(cfg)

        if args.get("checkpoint", None) is not None:
            classifier = LeNet.load_from_checkpoint(
                args.get("checkpoint"), map_location="cpu", **cp_overwrite
            )
        else:
            classifier = LeNet(**args)
    elif cfg["name"].lower() == "blenet":
        args = dict(
            cl_dim=dataset["cfg"]["cl_dim"],
            image_channels=dataset["cfg"]["image_channels"],
            image_size=dataset["cfg"]["image_size"],
        )
        args.update(cfg)

        if args.get("checkpoint", None) is not None:
            classifier = BLeNet.load_from_checkpoint(
                args.get("checkpoint"), map_location="cpu", **cp_overwrite
            )
        else:
            classifier = BLeNet(**args)
    elif cfg["name"].lower() == "medium":
        args = dict(
            cl_dim=dataset["cfg"]["cl_dim"],
            image_channels=dataset["cfg"]["image_channels"],
            image_size=dataset["cfg"]["image_size"],
        )
        args.update(cfg)

        if args.get("checkpoint", None) is not None:
            classifier = MediumClassifier.load_from_checkpoint(
                args.get("checkpoint"), map_location="cpu", **cp_overwrite
            )
        else:
            classifier = MediumClassifier(**args)
    elif cfg["name"].lower() == "small":
        args = dict(
            cl_dim=dataset["cfg"]["cl_dim"],
            image_channels=dataset["cfg"]["image_channels"],
            image_size=dataset["cfg"]["image_size"],
        )
        args.update(cfg)

        if args.get("checkpoint", None) is not None:
            classifier = SmallClassifier.load_from_checkpoint(
                args.get("checkpoint"), map_location="cpu", **cp_overwrite
            )
        else:
            classifier = SmallClassifier(**args)
    elif cfg["name"].lower() == "toy":
        args = dict()
        args.update(cfg)

        if args.get("checkpoint", None) is not None:
            classifier = ToyClassifier.load_from_checkpoint(
                args.get("checkpoint"), map_location="cpu", **cp_overwrite
            )
        else:
            classifier = ToyClassifier(**args)
    else:
        _log.error("Unknown model '{}'".format(cfg["name"]))
        raise ValueError("Unknown model '{}'".format(cfg["name"]))

    return classifier
