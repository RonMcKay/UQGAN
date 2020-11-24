from sacred import Ingredient
import torch.nn as nn

from datasets import datasets
from utils import load_config_from_checkpoint

from .medium import MediumCAE
from .resnet import DeepResNetCAE, GoodResNetCAE, ResNetCAE
from .small import SmallCAE

cae_models = Ingredient("cae_model", ingredients=(datasets,))

CAE_MODELS = ("small", "medium", "resnet", "resnet50", "good", "deep")


@cae_models.config
def config():
    cfg = dict()
    cfg["name"] = "small"
    cfg["latent_dim"] = 16
    cfg["checkpoint"] = None

    if cfg["checkpoint"] is not None:
        # try loading a saved configuration
        config_update = load_config_from_checkpoint(checkpoint_path=cfg["checkpoint"])
        config_update.pop("checkpoint", None)
        cfg.update(config_update)

        del config_update


@cae_models.capture
def load_cae_model(cfg, dataset, _log, **kwargs) -> nn.Module:
    if cfg["name"].lower() == "small":
        args = dict(
            cl_dim=dataset["cfg"]["cl_dim"],
            image_channels=dataset["cfg"]["image_channels"],
            image_size=dataset["cfg"]["image_size"],
        )
        args.update(cfg)
        args.update(kwargs)

        if args.get("checkpoint", None) is not None:
            cae = SmallCAE.load_from_checkpoint(args["checkpoint"], map_location="cpu")
        else:
            cae = SmallCAE(**args)
    elif cfg["name"].lower() == "medium":
        args = dict(
            cl_dim=dataset["cfg"]["cl_dim"],
            image_channels=dataset["cfg"]["image_channels"],
            image_size=dataset["cfg"]["image_size"],
        )
        args.update(cfg)
        args.update(kwargs)

        if args.get("checkpoint", None) is not None:
            cae = MediumCAE.load_from_checkpoint(args["checkpoint"], map_location="cpu")
        else:
            cae = MediumCAE(**args)
    elif cfg["name"].lower() == "resnet":
        args = dict(
            cl_dim=dataset["cfg"]["cl_dim"],
            image_channels=dataset["cfg"]["image_channels"],
            image_size=dataset["cfg"]["image_size"],
        )
        args.update(cfg)
        args.update(kwargs)

        if args.get("checkpoint", None) is not None:
            cae = ResNetCAE.load_from_checkpoint(args["checkpoint"], map_location="cpu")
        else:
            cae = ResNetCAE(**args)
    elif cfg["name"].lower() == "resnet50":
        _log.warning(
            "The 'resnet50' model has been configured for an image size of 32x32. "
            "If you want to change this you have to adjust the 'blocks'"
        )
        args = dict(
            cl_dim=dataset["cfg"]["cl_dim"],
            image_channels=dataset["cfg"]["image_channels"],
            image_size=dataset["cfg"]["image_size"],
            n_featuremaps=64,
            blocks=[4, 5, 7],
        )
        args.update(cfg)
        args.update(kwargs)

        if args.get("checkpoint", None) is not None:
            cae = ResNetCAE.load_from_checkpoint(args["checkpoint"], map_location="cpu")
        else:
            cae = ResNetCAE(**args)
    elif cfg["name"].lower() == "good":
        args = dict(
            cl_dim=dataset["cfg"]["cl_dim"],
            image_channels=dataset["cfg"]["image_channels"],
            image_size=dataset["cfg"]["image_size"],
        )
        args.update(cfg)
        args.update(kwargs)

        if args.get("checkpoint", None) is not None:
            cae = GoodResNetCAE.load_from_checkpoint(
                args["checkpoint"], map_location="cpu"
            )
        else:
            cae = GoodResNetCAE(**args)
    elif cfg["name"].lower() == "deep":
        args = dict(
            cl_dim=dataset["cfg"]["cl_dim"],
            image_channels=dataset["cfg"]["image_channels"],
            image_size=dataset["cfg"]["image_size"],
        )
        args.update(cfg)
        args.update(kwargs)

        if args.get("checkpoint", None) is not None:
            cae = DeepResNetCAE.load_from_checkpoint(
                args["checkpoint"], map_location="cpu"
            )
        else:
            cae = DeepResNetCAE(**args)
    else:
        _log.error("Unknown model '{}'".format(cfg["name"]))
        raise ValueError("Unknown model '{}'".format(cfg["name"]))

    return cae
