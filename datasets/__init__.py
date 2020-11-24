from copy import deepcopy
from functools import partial
import logging
from typing import Dict, Optional, Tuple

import numpy as np
from sacred import Ingredient
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import torchvision.transforms as trans
import torchvision.transforms.functional as ftrans
from typing_extensions import TypedDict

from datasets.celeba import CelebA
from datasets.cifar10 import CIFAR10
from datasets.cifar100 import CIFAR100
from datasets.classes_subset import ClassSubset
from datasets.clawa2 import ClAwA2  # noqa: F401
from datasets.datahandlers.cl import CUB200, AwA2
from datasets.emnist import EMNIST
from datasets.fmnist import FMNIST
from datasets.lsun import LSUN
from datasets.mnist import MNIST
from datasets.noise import NoiseImageDataset
from datasets.omniglot import Omniglot
from datasets.svhn import SVHN
from datasets.tinyimagenet import TinyImageNet
from datasets.toy import ToyDataset, ToyDataset2, ToyDataset3, ToyDataset4, ToyDataset5
from utils import IncompatibleRange, get_range

datasets = Ingredient("dataset")


class DatasetConfig(TypedDict, total=False):
    name: str
    image_size: int
    image_channels: int
    cl_dim: int
    mode: str
    static: bool


default_configs: Dict[str, DatasetConfig]
default_configs = dict(
    awa2=DatasetConfig(
        name="awa2",
        image_size=64,
        image_channels=3,
        cl_dim=85,
        mode="train",
        static=False,
    ),
    celeba=DatasetConfig(
        name="celeba",
        image_size=64,
        image_channels=3,
        cl_dim=40,
        mode="train",
        static=False,
    ),
    cifar10=DatasetConfig(
        name="cifar10",
        image_size=32,
        image_channels=3,
        cl_dim=10,
        mode="train",
        static=False,
    ),
    cifar100=DatasetConfig(
        name="cifar100",
        image_size=32,
        image_channels=3,
        cl_dim=100,
        mode="train",
        static=False,
    ),
    cub200=DatasetConfig(
        name="cub200",
        image_size=64,
        image_channels=3,
        cl_dim=312,
        static=False,
    ),
    emnist=DatasetConfig(
        name="emnist",
        image_size=32,
        image_channels=1,
        cl_dim=10,
        mode="train",
        static=False,
    ),
    emnist_l=DatasetConfig(
        name="emnist_l",
        image_size=32,
        image_channels=1,
        cl_dim=26,
        mode="train",
        static=False,
    ),
    fmnist=DatasetConfig(
        name="fmnist",
        image_size=32,
        image_channels=1,
        cl_dim=10,
        mode="train",
        static=False,
    ),
    lsun=DatasetConfig(
        name="lsun",
        image_size=32,
        image_channels=3,
        cl_dim=10,
        mode="train",
        static=False,
    ),
    mnist=DatasetConfig(
        name="mnist",
        image_size=32,
        image_channels=1,
        cl_dim=10,
        mode="train",
        static=False,
    ),
    normal_noise=DatasetConfig(
        name="normal_noise",
        image_size=32,
        image_channels=1,
        cl_dim=10,
    ),
    omniglot=DatasetConfig(
        name="omniglot",
        image_size=32,
        image_channels=1,
        cl_dim=964,
        mode="train",
        static=False,
    ),
    svhn=DatasetConfig(
        name="svhn",
        image_size=32,
        image_channels=3,
        cl_dim=10,
        mode="train",
        static=False,
    ),
    tinyimagenet=DatasetConfig(
        name="tinyimagenet",
        image_size=64,
        image_channels=3,
        cl_dim=200,
        mode="train",
        static=False,
    ),
    toy=DatasetConfig(
        name="toy",
        input_size=2,
        n_samples=100000,
        cl_dim=ToyDataset.CL_DIM,
    ),
    toy2=DatasetConfig(
        name="toy2",
        input_size=2,
        n_samples=100000,
        cl_dim=ToyDataset2.CL_DIM,
    ),
    toy3=DatasetConfig(
        name="toy3",
        input_size=2,
        n_samples=100000,
        cl_dim=ToyDataset3.CL_DIM,
    ),
    toy4=DatasetConfig(
        name="toy4",
        input_size=2,
        n_samples=100000,
        cl_dim=ToyDataset4.CL_DIM,
    ),
    toy5=DatasetConfig(
        name="toy5",
        input_size=2,
        n_samples=100000,
        cl_dim=ToyDataset5.CL_DIM,
    ),
    uniform_noise=DatasetConfig(
        name="uniform_noise",
        image_size=32,
        image_channels=1,
        cl_dim=10,
    ),
)


def format_dataset_config_str(
    dataset_config_str: str,
) -> Tuple[str, Optional[Tuple[int, ...]]]:
    name = dataset_config_str
    requested_classes = None
    if "+" in dataset_config_str:
        requested_classes = get_range(dataset_config_str.split("+")[-1])
        name = dataset_config_str.split("+")[0]

    return name, requested_classes


@datasets.config
def dataset_config():
    cfg = dict()
    cfg["name"] = "mnist"

    try:
        name, requested_classes = format_dataset_config_str(cfg["name"])
    except IncompatibleRange:
        raise ValueError(
            (
                f"Invalid class specification '{cfg['name'].split('+')[-1]}'.\n "
                "Should be a comma seperated list of '<int>' or '<int>-<int>'"
            )
        )

    try:
        cfg.update(default_configs[name])
    except KeyError:
        raise ValueError("Unknown dataset '{}'".format(name))

    if requested_classes is not None:
        cfg["cl_dim"] = len(requested_classes)

    del name, requested_classes


@datasets.capture
def load_data(
    cfg: DatasetConfig, _log: logging.Logger, **kwargs
) -> Tuple[Dataset, torch.utils.data.Sampler]:
    name, requested_classes = format_dataset_config_str(kwargs.get("name", cfg["name"]))

    if name not in default_configs:
        raise ValueError(f"Unknown dataset {name}")

    cfg = deepcopy(cfg)
    cfg.update(kwargs)
    cfg["_log"] = _log

    if name == "awa2":
        args = dict(split="train")
        args.update(cfg)
        transformations = []
        if not cfg.get("static", False):
            transformations.append(trans.RandomHorizontalFlip())

        transformations.extend(
            [
                trans.Resize(args["image_size"]),
                trans.CenterCrop(args["image_size"]),
            ]
        )
        if cfg.get("image_channels", None) == 1:
            transformations.append(trans.Grayscale())
        transformations.append(trans.ToTensor())

        traindat = AwA2(transform=trans.Compose(transformations), **args)
        weights = np.unique(traindat.img_index, return_counts=True)
        class_to_trainclass = {c: i for i, c in enumerate(weights[0])}
        weights = 1.0 / weights[1]
        sample_weights = weights[
            np.vectorize(class_to_trainclass.get)(traindat.img_index)
        ]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(traindat))
        if requested_classes is not None:
            traindat = ClassSubset(dataset=traindat, class_list=requested_classes)
            sampler = WeightedRandomSampler(
                sample_weights[traindat.indices], num_samples=len(traindat)
            )
    elif name == "celeba":
        args = dict()
        args.update(cfg)
        transformations = []
        if not cfg.get("static", False):
            transformations.append(trans.RandomHorizontalFlip())

        transformations.extend(
            [
                partial(ftrans.crop, top=40, left=20, height=218 - 80, width=178 - 40),
                trans.Resize(args["image_size"]),
            ]
        )
        if cfg.get("image_channels", None) == 1:
            transformations.append(trans.Grayscale())
        transformations.append(trans.ToTensor())

        traindat = CelebA(transform=trans.Compose(transformations), **args)
        if requested_classes is not None:
            traindat = ClassSubset(dataset=traindat, class_list=requested_classes)
        sampler = None
    elif name == "cifar10":
        args = dict()
        args.update(cfg)
        transformations = []
        if not cfg.get("static", False):
            transformations.append(trans.RandomHorizontalFlip())

        transformations.extend(
            [
                trans.Resize(args["image_size"]),
            ]
        )
        if cfg.get("image_channels", None) == 1:
            transformations.append(trans.Grayscale())
        transformations.append(trans.ToTensor())

        traindat = CIFAR10(transform=trans.Compose(transformations), **args)
        if requested_classes is not None:
            traindat = ClassSubset(dataset=traindat, class_list=requested_classes)
        sampler = None
    elif name == "cifar100":
        args = dict()
        args.update(cfg)
        transformations = []
        if not cfg.get("static", False):
            transformations.append(trans.RandomHorizontalFlip())

        transformations.extend(
            [
                trans.Resize(args["image_size"]),
            ]
        )
        if cfg.get("image_channels", None) == 1:
            transformations.append(trans.Grayscale())
        transformations.append(trans.ToTensor())

        traindat = CIFAR100(transform=trans.Compose(transformations), **args)
        if requested_classes is not None:
            traindat = ClassSubset(dataset=traindat, class_list=requested_classes)
        sampler = None
    elif name == "cub200":
        args = dict()
        args.update(cfg)
        transformations = []
        if not cfg.get("static", False):
            transformations.append(trans.RandomHorizontalFlip())

        transformations = [
            trans.Resize(args["image_size"]),
            trans.CenterCrop(args["image_size"]),
        ]
        if cfg.get("image_channels", None) == 1:
            transformations.append(trans.Grayscale())
        transformations.append(trans.ToTensor())

        traindat = CUB200(transform=trans.Compose(transformations), **args)
        if requested_classes is not None:
            traindat = ClassSubset(dataset=traindat, class_list=requested_classes)
        sampler = None
    elif name.startswith("emnist"):
        args = dict(
            download=True,
        )

        args["split"] = {"d": "digits", "l": "letters"}.get(
            name.split("_")[-1], "digits"
        )

        args.update(cfg)

        transformations = [
            trans.Resize(args["image_size"]),
            trans.Lambda(lambda img: ftrans.hflip(img)),
            trans.Lambda(lambda img: ftrans.rotate(img, 90)),
        ]
        transformations.append(trans.ToTensor())
        if cfg.get("image_channels", None) == 3:
            transformations.append(
                trans.Lambda(lambda img: torch.cat([img, img, img], dim=0))
            )

        traindat = EMNIST(transform=trans.Compose(transformations), **args)
        if requested_classes is not None:
            traindat = ClassSubset(dataset=traindat, class_list=requested_classes)
        sampler = None
    elif name == "fmnist":
        args = dict(
            download=True,
        )
        args.update(cfg)

        transformations = [
            trans.Resize(args["image_size"]),
        ]
        transformations.append(trans.ToTensor())
        if cfg.get("image_channels", None) == 3:
            transformations.append(
                trans.Lambda(lambda img: torch.cat([img, img, img], dim=0))
            )

        traindat = FMNIST(transform=trans.Compose(transformations), **args)
        if requested_classes is not None:
            traindat = ClassSubset(dataset=traindat, class_list=requested_classes)
        sampler = None
    elif name == "lsun":
        args = dict()
        args.update(cfg)
        transformations = []
        if not cfg.get("static", False):
            transformations.append(trans.RandomHorizontalFlip())

        transformations.extend(
            [
                trans.Resize(args["image_size"]),
                trans.CenterCrop(args["image_size"]),
            ]
        )
        if cfg.get("image_channels", None) == 1:
            transformations.append(trans.Grayscale())
        transformations.append(trans.ToTensor())

        traindat = LSUN(transform=trans.Compose(transformations), **args)
        if requested_classes is not None:
            traindat = ClassSubset(dataset=traindat, class_list=requested_classes)
        sampler = None
    elif name == "mnist":
        args = dict(download=True)
        args.update(cfg)

        transformations = [
            trans.Resize(args["image_size"]),
            trans.ToTensor(),
        ]
        if cfg.get("image_channels", None) == 3:
            transformations.append(
                trans.Lambda(lambda img: torch.cat([img, img, img], dim=0))
            )

        traindat = MNIST(transform=trans.Compose(transformations), **args)
        if requested_classes is not None:
            traindat = ClassSubset(dataset=traindat, class_list=requested_classes)
        sampler = None
    elif name == "omniglot":
        args = dict()
        args.update(cfg)

        transformations = [
            trans.Resize(args["image_size"]),
        ]
        transformations.append(trans.ToTensor())
        transformations.append(trans.Lambda(lambda img: 1 - img))
        if cfg.get("image_channels", None) == 3:
            transformations.append(
                trans.Lambda(lambda img: torch.cat([img, img, img], dim=0))
            )
        traindat = Omniglot(transform=trans.Compose(transformations), **args)
        if requested_classes is not None:
            traindat = ClassSubset(dataset=traindat, class_list=requested_classes)
        sampler = None
    elif name == "svhn":
        args = dict(
            split="train",
            download=True,
        )
        args.update(cfg)

        transformations = [trans.Resize(args["image_size"])]
        if cfg.get("image_channels", None) == 1:
            transformations.append(trans.Grayscale())
        transformations.append(trans.ToTensor())

        traindat = SVHN(transform=trans.Compose(transformations), **args)
        if requested_classes is not None:
            traindat = ClassSubset(dataset=traindat, class_list=requested_classes)
        sampler = None
    elif name == "tinyimagenet":
        args = dict()
        args.update(cfg)

        transformations = [trans.Resize(args["image_size"])]
        if cfg.get("image_channels", None) == 1:
            transformations.append(trans.Grayscale())
        transformations.append(trans.ToTensor())

        traindat = TinyImageNet(transform=trans.Compose(transformations), **args)
        if requested_classes is not None:
            traindat = ClassSubset(dataset=traindat, class_list=requested_classes)
        sampler = None
    elif name.startswith("toy"):
        args = dict()
        args.update(cfg)
        if name.endswith("y"):
            traindat = ToyDataset(**args)
        elif name.endswith("2"):
            traindat = ToyDataset2(**args)
        elif name.endswith("3"):
            traindat = ToyDataset3(**args)
        elif name.endswith("4"):
            traindat = ToyDataset4(**args)
        elif name.endswith("5"):
            traindat = ToyDataset5(**args)
        if requested_classes is not None:
            traindat = ClassSubset(dataset=traindat, class_list=requested_classes)
        sampler = None
    elif name in ("uniform_noise", "normal_noise"):
        args = dict(
            size=(cfg["image_channels"], cfg["image_size"], cfg["image_size"]),
            sample_type=cfg["name"].split("_")[0],
        )
        args.update(cfg)
        traindat = NoiseImageDataset(transform=trans.ToTensor(), **args)
        if requested_classes is not None:
            traindat = ClassSubset(dataset=traindat, class_list=requested_classes)
        sampler = None
    else:
        raise ValueError("Unknown Dataset '{}'".format(cfg["name"]))

    return traindat, sampler


if __name__ == "__main__":
    celeba_config = default_configs["celeba"]

    dat, _ = load_data(
        celeba_config,
        name="emnist+0",
        mode="eval",
        static=False,
        image_size=32,
    )

    dat2, _ = load_data(
        celeba_config,
        name="emnist",
        mode="eval",
        static=False,
        image_size=32,
    )

    print(f"Filtered size: {len(dat)}")
    print(f"Original size: {len(dat2)}")
