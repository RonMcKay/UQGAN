import datetime
import logging
import math
from math import ceil, sqrt
import os
from os.path import basename, exists, join, normpath
import re
import shutil
import sys
import traceback
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.run import Run
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd

from config import Config

log = logging.getLogger("utils")

AVAILABLE_REG_TYPES = (
    "cosine",
    "logcosine",
    "inversecosine",
    "abs",
    "euclid",
    "max",
    "min",
)


def init_weights(m: nn.Module) -> None:
    if isinstance(m, _ConvNd):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, _BatchNorm) or isinstance(m, nn.LayerNorm):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def init_gan_weights(m: nn.Module) -> None:
    if isinstance(m, _ConvNd):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        # nn.init.uniform_(m.weight, a=-1.0, b=1.0)
    elif isinstance(m, _BatchNorm):
        nn.init.normal_(m.weight, mean=1.0, std=0.02)
    else:
        if hasattr(m, "weight"):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            # nn.init.uniform_(m.weight, a=-1.0, b=1.0)

    if hasattr(m, "bias"):
        nn.init.constant_(m.bias, 0.0)


def init_experiment(ex: Experiment, mongo_observer: bool = True) -> None:
    log = logging.getLogger("root")
    log.handlers = []

    log_format = logging.Formatter(
        "[{levelname:.1s}] {asctime} || {name} - {message}", style="{"
    )

    streamhandler = logging.StreamHandler(sys.stdout)
    streamhandler.setFormatter(log_format)
    log.addHandler(streamhandler)

    log.setLevel(Config.log_level)
    ex.logger = log

    logging.getLogger("pytorch_lightning").setLevel(Config.log_level)

    if mongo_observer:
        if all(
            [
                hasattr(Config, c)
                for c in (
                    "mongo_url",
                    "mongo_db_name",
                    "mongo_username",
                    "mongo_password_file",
                )
            ]
        ):
            try:
                mobs = MongoObserver(
                    url=Config.mongo_url,
                    db_name=Config.mongo_db_name,
                    username=Config.mongo_username,
                    password=open(Config.mongo_password_file).read().replace("\n", ""),
                    authSource="admin",
                )
                ex.observers.append(mobs)
            except Exception:
                log.exception(
                    "Not able to add the configured MongoObserver! "
                    "I will proceed without it."
                )  # type: ignore
        else:
            log.info(
                "No MongoObserver configured. Some config is missing. "
                "See 'config.py' if you want to configure a MongoObserver"
            )


def get_experiment_folder(
    save_folder: str = Config.root_save_folder, _run: Optional[Run] = None
) -> str:
    id = None
    if _run is not None:
        # If an observer like e.g. the MongoObserver is supplying the experiment id,
        # use that one
        try:
            id = _run._id
        except AttributeError:
            pass

    if id is None and exists(save_folder):
        # else, find maximum experiment id in folder
        max_id = max(
            [0]
            + [
                extract_exp_id_from_path(join(save_folder, i))
                for i in os.listdir(save_folder)
                if is_experiment_folder(join(save_folder, i))
            ]
        )
        id = max_id + 1

        return join(save_folder, "experiment_{}".format(id))

    return None


def register_exp_folder(root_folder: str, _run: Optional[Run] = None) -> str:
    # Get the experiment folder by id and create it
    exp_folder = get_experiment_folder(root_folder, _run)
    if not exists(exp_folder):
        os.makedirs(exp_folder)
    elif getattr(_run, "_id", None) is not None:
        # Then the experiment got deleted from the mongodb and
        # can also be deleted from disk.
        # The folder has to be recreated.
        shutil.rmtree(exp_folder)
        os.makedirs(exp_folder)

    return exp_folder


def get_accelerator_device(
    gpus: Optional[Union[List[int], Tuple[int, ...], int]] = None
) -> Union[Tuple[None, None], Tuple[str, List[int]]]:
    if torch.cuda.is_available() and gpus is not None:
        accelerator = "gpu"
        devices = list(gpus) if isinstance(gpus, (list, tuple)) else [gpus]
    else:
        accelerator = None
        devices = None

    return accelerator, devices


class IncompatibleRange(Exception):
    pass


def get_range(inp: str) -> List[int]:
    matches = set()
    for i in inp.split(","):
        # match for a specified range
        match = re.match(r"(\d+)(?:\.{2,}|-)(\d+)", i)
        if match is not None:
            start = int(match.group(1))
            end = int(match.group(2)) + 1
            matches |= set(range(start, end))
        else:
            # If there was no range then it should be a single number
            try:
                matches.add(int(i))
            except ValueError:
                raise IncompatibleRange
    matches = list(matches)
    matches.sort()
    return matches


def format_int_list(list_of_ints: List[int]) -> str:
    if len(list_of_ints) == 1:
        return str(list_of_ints[0])
    list_of_ints.sort()
    consecutive = True
    for i in range(len(list_of_ints) - 1):
        if (list_of_ints[i + 1] - list_of_ints[i]) != 1:
            consecutive = False
            break

    if consecutive:
        formatted_list_of_ints = f"{list_of_ints[0]}-{list_of_ints[-1]}"
    else:
        formatted_list_of_ints = "_".join(str(i) for i in list_of_ints)

    return formatted_list_of_ints


def is_experiment_folder(path: str) -> bool:
    return re.match(
        r"experiment_\d+", basename(normpath(path))
    ) is not None and os.path.isdir(path)


def extract_exp_id_from_path(path: str) -> Union[int, None]:
    match = re.match(f".*{os.sep}experiment_(\\d+)(?:{os.sep}.*|$)", path)
    if match is not None:
        return int(match.group(1))
    return None


def cosine_loss_classwise(
    x: torch.Tensor,
    x_tilde: torch.Tensor,
    classes: torch.Tensor,
    space: str = "cos",
    eps: float = 1e-6,
) -> torch.Tensor:
    ang_total = 0
    total_counts = 0
    cl, inverse_indices, cl_counts = torch.unique(
        classes, return_counts=True, return_inverse=True, dim=0
    )
    for i in range(cl.shape[0]):
        if cl_counts[i] > 1:
            class_mask = inverse_indices == i
            total_counts += cl_counts[i]
            for j in range(cl_counts[i]):
                x_tilde_norm = (x_tilde[class_mask] - x[class_mask][j]) / (
                    x_tilde[class_mask] - x[class_mask][j]
                ).pow(2).sum(-1, keepdim=True).sqrt()

                sim = torch.clamp(
                    torch.matmul(x_tilde_norm, x_tilde_norm.transpose(1, 0)),
                    -1.0 + eps,
                    1.0 - eps,
                )

                if space == "cos":
                    sim = 1 - (torch.acos(sim) / math.pi)
                elif space == "log":
                    sim = torch.log(torch.acos(sim) / math.pi + eps).neg()
                elif space == "inverse":
                    sim = math.pi / (torch.acos(sim) + eps)

                sim = (
                    torch.triu(sim, diagonal=1)
                    .sum()
                    .div((cl_counts[i] * (cl_counts[i] - 1)) / 2)
                )

                ang_total = ang_total + sim

    return ang_total.div(total_counts)


def p_norm_loss(
    x_tilde: torch.Tensor,
    classes: torch.Tensor,
    p: float = float("inf"),
    scale: float = 2,
    eps: float = 1e-16,
) -> torch.Tensor:
    # standardize points
    x_std = (x_tilde - x_tilde.mean(dim=0, keepdim=True)) / x_tilde.std(
        dim=0, keepdim=True
    )

    # compute distances
    dist_total = 0
    total_count = 0
    cl, inverse_indices, cl_counts = torch.unique(
        classes, return_counts=True, return_inverse=True, dim=0
    )
    for i in range(cl.shape[0]):
        if cl_counts[i] >= 2:
            total_count += (cl_counts[i] * (cl_counts[i] - 1)) / 2
            class_mask = inverse_indices == i
            dist_total = (
                dist_total
                + torch.log(
                    torch.triu(
                        torch.cdist(x_std[class_mask], x_std[class_mask], p=p),
                        diagonal=1,
                    )
                    + eps
                )
                .neg()
                .div(scale)
                .sum()
            )

    return dist_total / total_count


def min_norm_loss(
    x_tilde: torch.Tensor,
    classes: torch.Tensor,
    eps: float = 1e-16,
) -> torch.Tensor:
    # standardize points
    x_std = (x_tilde - x_tilde.mean(dim=0, keepdim=True)) / x_tilde.std(
        dim=0, keepdim=True
    )

    # compute distances
    dist_total = 0
    total_count = 0
    cl, inverse_indices, cl_counts = torch.unique(
        classes, return_counts=True, return_inverse=True, dim=0
    )
    for i in range(cl.shape[0]):
        if cl_counts[i] >= 2:
            total_count += (cl_counts[i] * (cl_counts[i] - 1)) / 2
            class_mask = inverse_indices == i
            dist_total = (
                dist_total
                + torch.log(
                    (x_std[class_mask].unsqueeze(1) - x_std[class_mask]).abs() + eps
                )
                .neg()
                .max(2)[0]
                .triu(diagonal=1)
                .sum()
            )

    return dist_total / total_count


def entropy(x: torch.Tensor, dim: int = 1, eps: float = 1e-16) -> torch.Tensor:
    x = x + eps
    return (
        (x.log() * x)
        .sum(dim)
        .div(torch.log(torch.tensor(x.shape[dim], dtype=torch.float)))
        .neg()
    )


def load_config_from_checkpoint(checkpoint_path: str) -> dict:
    return dict(
        torch.load(checkpoint_path, map_location="cpu").get("hyper_parameters", {})
    )


def save_sample_images(
    folder: str,
    images: torch.Tensor,
    iteration: int,
    titles: Optional[Sequence[str]] = None,
    original_images: torch.Tensor = None,
) -> None:
    log = logging.getLogger("root.utils.save_samples")
    if len(images.shape) != 4:
        raise ValueError(
            "'images' need to have 4 dimensions, got {} [{}].".format(
                len(images.shape), tuple(images.shape)
            )
        )

    if images.shape[1] == 1:
        gray = True
    else:
        gray = False

    if original_images is not None:
        fig, ax = plt.subplots(nrows=images.shape[0], ncols=2, figsize=(7, 14))
        fig.suptitle("Reconstruction / Original")

        ax[0, 0].set_title("Reconstruction")
        ax[0, 1].set_title("Original")
    else:
        grid_size = ceil(sqrt(images.shape[0]))
        fig, ax = plt.subplots(nrows=grid_size, ncols=grid_size, figsize=(7, 7))

    for axis in ax.reshape(-1):
        axis.set_axis_off()

    log.debug("Saving Generator samples...")
    for i in range(images.shape[0]):
        if original_images is not None:
            ax1 = ax[i, 0]
            ax2 = ax[i, 1]
            ax2.set_title(titles[i] if titles is not None else "")
        else:
            ax1 = ax.reshape(-1)[i]
        ax1.set_title(titles[i] if titles is not None else "")

        if gray:
            ax1.imshow(images[i].squeeze(), cmap="gray")
            if original_images is not None:
                ax2.imshow(original_images[i].squeeze(), cmap="gray")
        else:
            ax1.imshow(images[i].permute(1, 2, 0))
            if original_images is not None:
                ax2.imshow(original_images[i].permute(1, 2, 0))

    # plt.axis("off")
    fig.savefig(
        join(
            folder,
            "{:0>{width}d}_sample_images.png".format(iteration, width=6),
        )
    )
    plt.close()


class TimeEstimator(Callback):
    def __init__(
        self,
        max_iterations: int,
        logger: Optional[logging.Logger] = None,
        interval: str = "epoch",
        divider: int = 1,
    ):
        if interval not in ("epoch", "step"):
            raise ValueError("'interval' has to be either epoch or step.")
        self.interval = interval
        self.max_iterations = max_iterations
        self.divider = divider
        if logger is None:
            self.logger = logging.getLogger("root.utils.TimeEstimator")
        else:
            self.logger = logger.getChild("TimeEstimator")

        self.start_time = None

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.start_time = datetime.datetime.now()
        self.start_epoch = 0

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if self.interval == "step":
            self.log_remaining((trainer.global_step + 1) // self.divider)
        else:
            self.log_remaining((trainer.current_epoch + 1) // self.divider)

    def log_remaining(self, epoch: int = None) -> None:
        if self.start_time is None:
            self.logger.warning(
                "TimeEstimator has been called before start. Starting now."
            )
            self.start()
            return
        elif epoch is None:
            # just print the duration if no epoch is specified
            current_time = datetime.datetime.now()
            duration = datetime.timedelta(
                seconds=int((current_time - self.start_time).total_seconds())
            )
            self.start_time = current_time
            self.logger.info("Iteration took {}".format(duration))
            return
        elif epoch == self.start_epoch:
            return
        elif epoch < self.max_iterations:
            duration = datetime.timedelta(
                seconds=int((datetime.datetime.now() - self.start_time).total_seconds())
            )
            remaining = datetime.timedelta(
                seconds=int(
                    duration.total_seconds()
                    / (epoch - self.start_epoch)
                    * (self.max_iterations - (epoch - self.start_epoch))
                )
            )
        elif epoch == self.max_iterations:
            return
        else:
            self.logger.warning(
                "Called TimeEstimator with an epoch greater than max_epochs!"
            )
            return

        self.logger.info(
            f"{epoch:d} / {self.max_iterations:d} "
            f"{'steps' if self.interval == 'step' else 'epochs'} "
            f"- Approximately {remaining} remaining"
        )
