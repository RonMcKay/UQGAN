from copy import deepcopy
from typing import Tuple

from sacred import Ingredient
import torch.nn as nn

from datasets import datasets
from utils import load_config_from_checkpoint

from .dcgan import Discriminator as DCGANDiscriminator
from .dcgan import Generator as DCGANGenerator
from .resnet import (
    DeepDiscriminator,
    DeepGenerator,
    Discriminator,
    Generator,
    GoodDiscriminator,
    GoodGenerator,
)
from .sensoyetal2020 import DiscriminatorImageSpace, DiscriminatorLatentSpace
from .sensoyetal2020 import Generator as SensoyGenerator
from .small import SmallDiscriminator, SmallGenerator
from .toy import (
    LargerToyDiscriminator,
    LargerToyGenerator,
    ToyDiscriminator,
    ToyGenerator,
)

gan_models = Ingredient("gan_model", ingredients=(datasets,))


GAN_MODELS = (
    "dcgan",
    "resnet",
    "deepresnet",
    "good",
    "small",
    "toy",
    "largertoy",
    "sensoyetal2020",
)


@gan_models.config
def config():
    cfg = dict()
    cfg["name"] = "toy"
    cfg["latent_dim"] = 128
    cfg["gen_checkpoint"] = None
    cfg["disc_checkpoint"] = None

    if cfg["name"].lower() == "sensoyetal2020":
        cfg["latent_dim"] = 100

    if cfg["name"].lower() in ("resnet", "deepresnet", "good"):
        cfg["n_featuremaps"] = 32
        cfg["discriminator_norm"] = "layernorm"

    if cfg["gen_checkpoint"] is not None:
        # try loading a saved configuration
        config_update = load_config_from_checkpoint(
            checkpoint_path=cfg["gen_checkpoint"]
        )
        config_update.pop("gen_checkpoint", None)
        config_update.pop("disc_checkpoint", None)
        cfg.update(config_update)

        del config_update

    if cfg["disc_checkpoint"] is not None:
        # try loading a saved configuration
        config_update = load_config_from_checkpoint(
            checkpoint_path=cfg["disc_checkpoint"]
        )
        config_update.pop("gen_checkpoint", None)
        config_update.pop("disc_checkpoint", None)
        cfg.update(config_update)

        del config_update


@gan_models.capture
def load_gan_model(cfg, dataset, _log, **kwargs) -> Tuple[nn.Module, nn.Module]:
    cfg = deepcopy(cfg)

    if "checkpoint" in kwargs and cfg.get("checkpoint", None) is None:
        cp_config_update = load_config_from_checkpoint(kwargs["checkpoint"])
        cfg.update(cp_config_update)

    cfg.update(kwargs)

    if cfg["name"].lower() == "dcgan":
        args = dict(
            cl_dim=dataset["cfg"]["cl_dim"],
            image_channels=dataset["cfg"]["image_channels"],
            image_size=dataset["cfg"]["image_size"],
        )
        args.update(cfg)

        if args.get("gen_checkpoint", None) is not None:
            generator = DCGANGenerator.load_from_checkpoint(
                args.get("gen_checkpoint"), map_location="cpu"
            )
        else:
            generator = DCGANGenerator(**args)

        if args.get("disc_checkpoint", None) is not None:
            discriminator = DCGANDiscriminator.load_from_checkpoint(
                args.get("disc_checkpoint"), map_location="cpu"
            )
        else:
            discriminator = DCGANDiscriminator(**args)
    elif cfg["name"].lower() == "resnet":
        args = dict(
            cl_dim=dataset["cfg"]["cl_dim"],
            image_channels=dataset["cfg"]["image_channels"],
            image_size=dataset["cfg"]["image_size"],
        )
        args.update(cfg)

        if args.get("gen_checkpoint", None) is not None:
            generator = Generator.load_from_checkpoint(
                args.get("gen_checkpoint"), map_location="cpu"
            )
        else:
            generator = Generator(**args)

        if args.get("disc_checkpoint", None) is not None:
            discriminator = Discriminator.load_from_checkpoint(
                args.get("disc_checkpoint"), map_location="cpu"
            )
        else:
            discriminator = Discriminator(**args)
    elif cfg["name"].lower() == "deepresnet":
        args = dict(
            cl_dim=dataset["cfg"]["cl_dim"],
            image_channels=dataset["cfg"]["image_channels"],
            image_size=dataset["cfg"]["image_size"],
        )
        args.update(cfg)

        if args.get("gen_checkpoint", None) is not None:
            generator = DeepGenerator.load_from_checkpoint(
                args.get("gen_checkpoint"), map_location="cpu"
            )
        else:
            generator = DeepGenerator(**args)

        if args.get("disc_checkpoint", None) is not None:
            discriminator = DeepDiscriminator.load_from_checkpoint(
                args.get("disc_checkpoint"), map_location="cpu"
            )
        else:
            discriminator = DeepDiscriminator(**args)
    elif cfg["name"].lower() == "good":
        args = dict(
            cl_dim=dataset["cfg"]["cl_dim"],
            image_channels=dataset["cfg"]["image_channels"],
            image_size=dataset["cfg"]["image_size"],
        )
        args.update(cfg)

        if args.get("gen_checkpoint", None) is not None:
            generator = GoodGenerator.load_from_checkpoint(
                args.get("gen_checkpoint"), map_location="cpu"
            )
        else:
            generator = GoodGenerator(**args)

        if args.get("disc_checkpoint", None) is not None:
            discriminator = GoodDiscriminator.load_from_checkpoint(
                args.get("disc_checkpoint"), map_location="cpu"
            )
        else:
            discriminator = GoodDiscriminator(**args)
    elif cfg["name"].lower() == "small":
        args = dict(
            cl_dim=dataset["cfg"]["cl_dim"],
            image_channels=dataset["cfg"]["image_channels"],
            image_size=dataset["cfg"]["image_size"],
        )
        args.update(cfg)

        if args.get("gen_checkpoint", None) is not None:
            generator = SmallGenerator.load_from_checkpoint(
                args.get("gen_checkpoint"), map_location="cpu"
            )
        else:
            generator = SmallGenerator(**args)

        if args.get("disc_checkpoint", None) is not None:
            discriminator = SmallDiscriminator.load_from_checkpoint(
                args.get("disc_checkpoint"), map_location="cpu"
            )
        else:
            discriminator = SmallDiscriminator(**args)
    elif cfg["name"].lower() == "toy":
        args = dict()
        args.update(cfg)

        if args.get("gen_checkpoint", None) is not None:
            generator = ToyGenerator.load_from_checkpoint(
                args.get("gen_checkpoint"), map_location="cpu"
            )
        else:
            generator = ToyGenerator(**args)

        if args.get("disc_checkpoint", None) is not None:
            discriminator = ToyDiscriminator.load_from_checkpoint(
                args.get("disc_checkpoint"), map_location="cpu"
            )
        else:
            discriminator = ToyDiscriminator(**args)
    elif cfg["name"].lower() == "largertoy":
        args = dict()
        args.update(cfg)

        if args.get("gen_checkpoint", None) is not None:
            generator = LargerToyGenerator.load_from_checkpoint(
                args.get("gen_checkpoint"), map_location="cpu"
            )
        else:
            generator = LargerToyGenerator(**args)

        if args.get("disc_checkpoint", None) is not None:
            discriminator = LargerToyDiscriminator.load_from_checkpoint(
                args.get("disc_checkpoint"), map_location="cpu"
            )
        else:
            discriminator = LargerToyDiscriminator(**args)
    elif cfg["name"].lower() == "sensoyetal2020":
        args = dict(
            cl_dim=dataset["cfg"]["cl_dim"],
            image_channels=dataset["cfg"]["image_channels"],
            image_size=dataset["cfg"]["image_size"],
        )
        args.update(cfg)

        generator = SensoyGenerator(**args)
        disc1 = DiscriminatorLatentSpace(**args)
        disc2 = DiscriminatorImageSpace(**args)

        return generator, disc1, disc2
    else:
        _log.error("Unknown model '{}'".format(cfg["name"]))
        raise ValueError("Unknown model '{}'".format(cfg["name"]))

    return generator, discriminator
