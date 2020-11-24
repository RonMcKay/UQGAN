from copy import deepcopy
import logging
from logging import Logger
from os.path import basename, dirname, join
import re
from typing import Any, Callable, Dict, Optional, Union

import matplotlib

from cls_models import set_model_to_mode
from cls_models.base import BaseClassifier
from datasets import load_data
from datasets.toy import ToyDataset, ToyDataset2, ToyDataset3

matplotlib.use("Agg")

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning import LightningModule
from pytorch_lightning.plugins.io import CheckpointIO
from pytorch_lightning.utilities.cloud_io import atomic_save, get_filesystem
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.types import _PATH
from sacred.config.config_dict import ConfigDict
from scipy.stats import gaussian_kde
import torch
from torch import optim
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as tf

from eval_ood_detection import eval_classifier
from utils import cosine_loss_classwise, min_norm_loss, p_norm_loss


class UQGAN(LightningModule):
    def __init__(
        self,
        classifier: BaseClassifier,
        generator: nn.Module,
        discriminator: nn.Module,
        cae: nn.Module,
        args: Dict,
        dataset: Dict,
        opt: Dict = {},
    ) -> None:
        super().__init__()

        self.classifier = classifier
        self.generator = generator
        self.discriminator = discriminator
        self.cae = cae
        for param in self.cae.parameters():
            param.requires_grad = False

        self.args = args
        self.dataset = dataset
        self.opt = opt

        self.save_hyperparameters(
            ignore=["classifier", "generator", "discriminator", "cae"]
        )

        self.class_crit = nn.BCELoss(reduction="none")

        self.automatic_optimization = False
        self.console_logger = logging.getLogger("root").getChild("UQGAN")

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        for module in ("classifier", "generator", "discriminator"):
            if hasattr(getattr(self, module), "_hparams_name"):
                checkpoint[f"{module}_hparams_name"] = getattr(
                    self, module
                )._hparams_name
                checkpoint[f"{module}_hyper_parameters"] = getattr(self, module).hparams

    def configure_optimizers(self):
        cls_opt = optim.Adam(
            self.classifier.parameters(),
            lr=self.opt.get("lr_cls", self.opt.get("lr", 1e-3)),
            weight_decay=self.opt.get(
                "weight_decay_cls", self.opt.get("weight_decay", 0)
            ),
        )
        cls_sched = optim.lr_scheduler.LinearLR(
            cls_opt,
            start_factor=1.0,
            end_factor=self.opt.get("min_lr_cls", self.opt.get("min_lr", 1e-5))
            / self.opt.get("lr_cls", self.opt.get("lr", 1e-3)),
            total_iters=self.args["iterations"],
        )

        gen_opt = optim.Adam(
            self.generator.parameters(),
            lr=self.opt.get("lr_gen", self.opt.get("lr", 2e-4)),
            betas=(0, 0.9),
            weight_decay=self.opt.get(
                "weight_decay_gen", self.opt.get("weight_decay", 0)
            ),
        )
        gen_sched = optim.lr_scheduler.LinearLR(
            gen_opt,
            start_factor=1.0,
            end_factor=self.opt.get("min_lr_gen", self.opt.get("min_lr", 1e-5))
            / self.opt.get("lr_gen", self.opt.get("lr", 2e-4)),
            total_iters=self.args["iterations"],
        )

        disc_opt = optim.Adam(
            self.discriminator.parameters(),
            lr=self.opt.get("lr_disc", self.opt.get("lr", 2e-4)),
            betas=(0, 0.9),
            weight_decay=self.opt.get(
                "weight_decay_disc", self.opt.get("weight_decay", 0)
            ),
        )
        disc_sched = optim.lr_scheduler.LinearLR(
            disc_opt,
            start_factor=1.0,
            end_factor=self.opt.get("min_lr_disc", self.opt.get("min_lr", 1e-5))
            / self.opt.get("lr_disc", self.opt.get("lr", 2e-4)),
            total_iters=self.args["iterations"],
        )

        return [disc_opt, gen_opt, cls_opt], [disc_sched, gen_sched, cls_sched]

    def compute_regloss(
        self,
        x_encoding: torch.Tensor,
        x_encoding_tilde: torch.Tensor,
        y_oh: torch.Tensor,
    ) -> torch.Tensor:
        regloss = torch.tensor(0.0, device=self.device)
        regtypes = self.args["reg_type"].split(",")
        for i, reg_t in enumerate(regtypes):
            weight = (
                1.0 if "reg_weight" not in self.args else self.args["reg_weight"][i]
            )

            if reg_t == "cosine":
                tmp_reg_loss = cosine_loss_classwise(
                    x_encoding, x_encoding_tilde, y_oh, space="cos"
                )
            elif reg_t == "logcosine":
                tmp_reg_loss = cosine_loss_classwise(
                    x_encoding, x_encoding_tilde, y_oh, space="log"
                )
            elif reg_t == "inversecosine":
                tmp_reg_loss = cosine_loss_classwise(
                    x_encoding, x_encoding_tilde, y_oh, space="inverse"
                )
            elif reg_t == "abs":
                tmp_reg_loss = p_norm_loss(x_encoding_tilde, y_oh, p=1)
            elif reg_t == "euclid":
                tmp_reg_loss = p_norm_loss(x_encoding_tilde, y_oh, p=2)
            elif reg_t == "max":
                tmp_reg_loss = p_norm_loss(x_encoding_tilde, y_oh)
            elif reg_t == "min":
                tmp_reg_loss = min_norm_loss(x_encoding_tilde, y_oh)

            regloss = regloss + tmp_reg_loss * weight
            self.log(f"{reg_t}_reg_loss", tmp_reg_loss.item())
            return regloss

    def training_step(self, batch: Any, batch_idx: int) -> None:
        x, y = batch
        y_oh = tf.one_hot(y, num_classes=self.dataset["cfg"]["cl_dim"]).to(
            torch.float32
        )
        self.cae.eval()

        ############################
        #    Train Discriminator
        ############################
        self.generator.eval()
        self.discriminator.train()
        set_model_to_mode(self.classifier, "eval")

        opt = self.optimizers()[0]

        with torch.no_grad():
            x_encoding = self.cae.encode(x, y_oh)
            x_encoding_tilde = self.generator(y_oh)
            x_tilde = self.cae.decode(x_encoding_tilde, y_oh)
            if "image_size" in self.dataset["cfg"]:
                x_tilde = torch.sigmoid(x_tilde)

        avg_disc_loss = 0
        avg_grad_loss = 0
        for _ in range(self.args.get("discriminator_iterations", 1)):
            opt.zero_grad()
            disc_x_tilde = self.discriminator(x_encoding_tilde, y_oh)
            disc_x = self.discriminator(x_encoding, y_oh)

            disc_loss = disc_x_tilde.mean() - disc_x.mean()  # type: torch.Tensor
            grad_loss = gradient_penalty(
                self.discriminator, x_encoding, x_encoding_tilde, y_oh
            )

            disc_loss = disc_loss + self.args["lambda_gp"] * grad_loss

            self.manual_backward(disc_loss)
            opt.step()

            avg_disc_loss += disc_loss.item()
            avg_grad_loss += grad_loss.item()

        self.log(
            "disc_loss", avg_disc_loss / self.args.get("discriminator_iterations", 1)
        )
        self.log(
            "disc_grad_loss",
            avg_grad_loss / self.args.get("discriminator_iterations", 1),
        )

        ############################
        #    Train Generator
        ############################
        self.generator.train()
        self.discriminator.eval()
        set_model_to_mode(self.classifier, "eval")

        opt = self.optimizers()[1]
        opt.zero_grad()

        x_encoding_tilde = self.generator(y_oh)
        x_tilde = self.cae.decode(x_encoding_tilde, y_oh)
        if "image_size" in self.dataset["cfg"]:
            x_tilde = torch.sigmoid(x_tilde)

        regloss = self.compute_regloss(x_encoding, x_encoding_tilde, y_oh)

        disc_x_tilde = self.discriminator(x_encoding_tilde, y_oh)

        class_x_tilde = self.classifier(x_tilde, return_ova_probs=True)[0]
        class_loss_x_tilde = self.class_crit(
            class_x_tilde, torch.zeros_like(class_x_tilde)
        )
        class_loss_x_tilde = torch.gather(
            class_loss_x_tilde, 1, y.long().view(-1, 1)
        ).mean()

        gen_loss = (
            disc_x_tilde.mean().neg()
            + self.args["lambda_cl_loss"] * class_loss_x_tilde
            + self.args["lambda_reg_loss"] * regloss
        )

        self.manual_backward(gen_loss)
        opt.step()

        self.log("gen_loss", gen_loss.item())

        ############################
        #    Train Classifier
        ############################
        self.generator.eval()
        self.discriminator.eval()
        set_model_to_mode(self.classifier, "train")

        opt = self.optimizers()[2]

        with torch.no_grad():
            x_encoding_tilde = self.generator(y_oh)
            x_tilde = self.cae.decode(x_encoding_tilde, y_oh)
            if "image_size" in self.dataset["cfg"]:
                x_tilde = torch.sigmoid(x_tilde)

        avg_class_loss = 0
        for _ in range(self.args.get("classifier_iterations", 1)):
            opt.zero_grad()
            class_x = self.classifier(x, return_ova_probs=True)[0]
            class_x_tilde = self.classifier(x_tilde, return_ova_probs=True)[0]

            in_dist_mask = tf.one_hot(y.long(), num_classes=class_x.shape[1])

            class_loss_x = self.class_crit(
                class_x, in_dist_mask.float()
            )  # type: torch.Tensor
            class_loss_x_one = class_loss_x[in_dist_mask == 1].sum().div(x.shape[0])
            class_loss_x_all = (
                class_loss_x[in_dist_mask == 0]
                * self.classifier.neg_weight.index_select(0, y.long())[
                    in_dist_mask == 0
                ]
            ).mean()

            class_loss_x_tilde = self.class_crit(
                class_x_tilde, torch.zeros_like(class_x_tilde)
            )
            class_loss_x_tilde = (
                torch.gather(class_loss_x_tilde, 1, y.long().view(-1, 1))
            ).mean()

            class_loss = (
                class_loss_x_one
                + self.args["lambda_real_ood"] * class_loss_x_all
                + (1 - self.args["lambda_real_ood"]) * class_loss_x_tilde
            )

            self.manual_backward(class_loss)
            opt.step()

            avg_class_loss += class_loss.item()

        self.log(
            "class_loss",
            avg_class_loss / self.args.get("classifier_iterations", 1),
        )

        for sched in self.lr_schedulers():
            sched.step()

    def validation_step(self, batch, batch_idx, **kwargs) -> None:
        set_model_to_mode(self.classifier, "eval")
        x, y = batch
        y_hat = self.classifier(x)[0].argmax(1)
        val_acc = (y_hat == y).float().mean().item()
        self.log("val_acc", val_acc)

    def on_train_epoch_end(self) -> None:
        if getattr(self, "max_acc", 0) < self.trainer.logged_metrics.get(
            "val_acc", 0
        ) and self.args.get("ood_datasets", ""):
            self.console_logger.info("Evaluating OOD-Detection...")
            self.max_acc = self.trainer.logged_metrics["val_acc"]

            eval_dataset_config = deepcopy(self.dataset["cfg"])
            eval_dataset_config["mode"] = "eval"
            eval_dataset_config["static"] = True

            indist_dataset = load_data(**eval_dataset_config)[0]
            self.console_logger.debug(
                f"{len(indist_dataset)} In-Dist Samples of "
                f"{eval_dataset_config['name']}"
            )

            ood_datasets = []
            if self.args["ood_datasets"] is None:
                raise ValueError("No OOD dataset specified")
            else:
                for d in self.args["ood_datasets"].split(","):
                    ood_cfg = dict(
                        name=d,
                        mode=eval_dataset_config["mode"],
                        static=True,
                        image_channels=eval_dataset_config["image_channels"],
                    )
                    if "image_size" in eval_dataset_config:
                        ood_cfg["image_size"] = eval_dataset_config["image_size"]
                    ood_datasets.append(load_data(**ood_cfg)[0])
                    self.console_logger.debug(
                        f"{len(ood_datasets[-1])} OOD-Samples of '{d}'"
                    )

            set_model_to_mode(self.classifier, "eval")
            result = eval_classifier(
                classifier=self.classifier,
                indist_dataset=indist_dataset,
                ood_datasets=ood_datasets,
                args=self.args,
                log=self.console_logger,
            )[0]

            for ood_index, ood_name in enumerate(self.args["ood_datasets"].split(",")):
                for metric_index, metric_name in enumerate(
                    ["auroc", "aupr-in", "aupr-out", "fpr95tpr"]
                ):
                    self.log(
                        f"{metric_name}-{ood_name}", result[ood_index, metric_index]
                    )
            self.log("auroc-all", result[-2, 0])
            self.log("aupr-in-all", result[-2, 1])
            self.log("aupr-out-all", result[-2, 2])
            self.log("fpr95tpr-all", result[-2, 3])

            self.log("auroc-succ/fail", result[-1, 0])
            self.log("aupr-s-succ/fail", result[-1, 1])
            self.log("aupr-f-succ/fail", result[-1, 2])
            self.log("fpr95tpr-succ/fail", result[-1, 3])


def gradient_penalty(
    discriminator: nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    real_y_oh: torch.Tensor,
) -> torch.Tensor:
    interpolation_factor_shape = (real_samples.shape[0],) + (1,) * (
        len(real_samples.shape) - 1
    )
    interpolation_factor = torch.rand(interpolation_factor_shape).to(
        real_samples.device
    )
    interpolations = (
        interpolation_factor * real_samples + (1 - interpolation_factor) * fake_samples
    ).requires_grad_(True)
    disc_out_interpolations = discriminator(interpolations, real_y_oh)

    gradients = autograd.grad(
        outputs=disc_out_interpolations,
        inputs=interpolations,
        grad_outputs=torch.full_like(
            disc_out_interpolations,
            1.0,
            device=real_samples.device,
            requires_grad=False,
        ),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.shape[0], -1)
    return torch.mean((gradients.norm(2, dim=1) - 1) ** 2)


class CustomCheckpointIO(CheckpointIO):
    def __init__(self, modules=("classifier", "generator", "discriminator")) -> None:
        super().__init__()

        self.modules = modules

    @staticmethod
    def get_path_template(path):
        m = re.match(r"(?:(?!(?:-v\d+|last)\.).)*-?(v\d+|last)?\.(.*)", basename(path))
        if m.group(1) is not None:
            return join(dirname(path), f"{{}}-{m.group(1)}.{m.group(2)}")
        else:
            return join(dirname(path), f"{{}}.{m.group(2)}")

    @staticmethod
    def remove_version_from_path(path: str) -> str:
        return join(dirname(path), re.sub(r"-v\d+\.", ".", basename(path)))

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: _PATH,
        storage_options: Optional[Any] = None,
    ) -> None:
        path = self.remove_version_from_path(path)
        fs = get_filesystem(path)
        fs.makedirs(dirname(path), exist_ok=True)

        for module in self.modules:
            cp_part = {}
            cp_part["state_dict"] = {
                k[len(module + ".") :]: v
                for k, v in checkpoint["state_dict"].items()
                if k.startswith(module + ".")
            }
            if f"{module}_hparams_name" in checkpoint:
                cp_part["hparams_name"] = checkpoint.get(f"{module}_hparams_name", {})
                cp_part["hyper_parameters"] = checkpoint.get(
                    f"{module}_hyper_parameters", {}
                )

            filename = join(dirname(path), self.get_path_template(path).format(module))

            atomic_save(cp_part, filename)

        # atomic_save(checkpoint, path)

    def load_checkpoint(
        self,
        path: _PATH,
        map_location: Optional[Callable] = lambda storage, loc: storage,
    ) -> Dict[str, Any]:
        fs = get_filesystem(path)
        if fs.exists(path):
            return pl_load(path, map_location=map_location)

    def remove_checkpoint(self, path: _PATH) -> None:
        fs = get_filesystem(path)
        if fs.exists(path):
            fs.rm(path, recursive=True)


def plot_toy(
    in_distribution_data: torch.Tensor,
    out_of_distribution_data: torch.Tensor,
    toydataset: Union[ToyDataset, ToyDataset2, ToyDataset3],
    y_oh: torch.Tensor,
    classifier: BaseClassifier,
    discriminator: nn.Module,
    iteration: int,
    save_folder: str,
    _log: Logger,
    args: ConfigDict = None,
    max_visual_points: int = 100,
) -> None:
    global xymin, xymax
    cm = plt.get_cmap("nipy_spectral")

    c = np.unique(y_oh, axis=0, return_inverse=True)[1]
    c = c - c.min()
    c = c / c.max()

    y_colors = cm(c)

    if xymin is None or xymax is None:
        scale = 1.3
        xmin = in_distribution_data[:, 0].min().numpy()
        xmax = in_distribution_data[:, 0].max().numpy()
        ymin = in_distribution_data[:, 1].min().numpy()
        ymax = in_distribution_data[:, 1].max().numpy()
        xymin = min(
            xmin - abs(xmax - xmin) * max(scale - 1, 0),
            ymin - abs(ymax - ymin) * max(scale - 1, 0),
        )
        xymax = max(
            xmax + abs(xmax - xmin) * max(scale - 1, 0),
            ymax + abs(ymax - ymin) * max(scale - 1, 0),
        )

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        ax.set_xlim(left=xymin, right=xymax)
        ax.set_ylim(bottom=xymin, top=xymax)
        ax.scatter(
            in_distribution_data[:max_visual_points, 0],
            in_distribution_data[:max_visual_points, 1],
            c=y_colors[: in_distribution_data.shape[0]][:max_visual_points],
        )

        if i < len(axes.flatten()) - 1:
            ax.scatter(
                out_of_distribution_data[:max_visual_points, 0],
                out_of_distribution_data[:max_visual_points, 1],
                c=y_colors[:max_visual_points],
                edgecolor="r",
                alpha=0.6,
            )

    # build grid
    density = 100
    grid_x, grid_y = np.mgrid[
        xymin : xymax : complex(density), xymin : xymax : complex(density)
    ]  # type: ignore
    heatmap_points = torch.cat(
        (
            torch.tensor(grid_x.flatten()).unsqueeze(-1),
            torch.tensor(grid_y.flatten()).unsqueeze(-1),
        ),
        dim=1,
    ).float()

    # Plot discriminator decision boundary
    if hasattr(toydataset, "construct_onehot"):
        _log.debug("Computing discriminator decision boundary...")
        heatmap_onehot = toydataset.construct_onehot(heatmap_points)
        heatmap_values_disc = torch.empty((0,))
        with torch.no_grad():  # type: ignore
            for x, onehot in zip(
                heatmap_points.split(split_size=256, dim=0),  # type: ignore
                heatmap_onehot.split(split_size=256, dim=0),  # type: ignore
            ):
                heatmap_values_disc = torch.cat(
                    (heatmap_values_disc, discriminator(x, onehot).cpu())
                )

        _log.debug(
            (
                f"'heatmap_values_disc value range: [{heatmap_values_disc.min()}, "
                f"{heatmap_values_disc.max()}]'"
            )
        )
        heatmap_values_disc = (heatmap_values_disc - heatmap_values_disc.min()) / (
            heatmap_values_disc.max() - heatmap_values_disc.min()
        )

        axes[0, 0].set_title("discriminator decision boundary")
        axes[0, 0].pcolormesh(
            grid_x,
            grid_y,
            heatmap_values_disc.numpy().reshape(grid_x.shape),
            cmap=plt.get_cmap("RdYlGn"),
            shading="gouraud",
            zorder=0,
        )

    # Plot classifier outputs
    _log.debug("Computing classifier heatmaps")
    heatmap_au = torch.empty((0,))
    heatmap_eu = torch.empty((0,))
    with torch.no_grad():  # type: ignore
        for x in heatmap_points.split(split_size=256, dim=0):  # type: ignore
            x = x

            cls_out = classifier(x)

            heatmap_au = torch.cat((heatmap_au, cls_out[1]))
            heatmap_eu = torch.cat((heatmap_eu, cls_out[2]))

    _log.debug(f"'heatmap_eu value range: [{heatmap_eu.min()}, {heatmap_eu.max()}]'")
    heatmap_eu = (heatmap_eu - heatmap_eu.min()) / (heatmap_eu.max() - heatmap_eu.min())
    heatmap_eu = 1 - heatmap_eu

    _log.debug(f"'heatmap_au value range: [{heatmap_au.min()}, {heatmap_au.max()}]'")
    heatmap_au = (heatmap_au - heatmap_au.min()) / (heatmap_au.max() - heatmap_au.min())
    heatmap_au = 1 - heatmap_au

    axes[0, 1].set_title("classifier OOD Heatmap")
    axes[0, 1].pcolormesh(
        grid_x,
        grid_y,
        heatmap_eu.numpy().reshape(grid_x.shape),
        cmap=plt.get_cmap("RdYlGn"),
        shading="gouraud",
        zorder=0,
    )

    axes[1, 0].set_title("classifier aleatoric uncertainty")
    axes[1, 0].pcolormesh(
        grid_x,
        grid_y,
        heatmap_au.numpy().reshape(grid_x.shape),
        cmap=plt.get_cmap("RdYlGn"),
        shading="gouraud",
        zorder=0,
    )

    # plot estimated density of generated samples
    embedding_kde = gaussian_kde(out_of_distribution_data.numpy().T)
    grid_density = embedding_kde(np.vstack([grid_x.flatten(), grid_y.flatten()]))
    colmap = plt.get_cmap("Blues")
    colmap = colors.LinearSegmentedColormap.from_list(
        f"trunc({colmap.name},{0.0:.2f},{1.0:.2f})", colmap(np.linspace(0.0, 1.0, 256))
    )
    grid_density[grid_density < np.quantile(grid_density, 0.55)] = np.NaN
    colmap.set_bad("white")

    axes[1, 1].set_title("Density of generated points")
    axes[1, 1].pcolormesh(
        grid_x,
        grid_y,
        grid_density.reshape(grid_x.shape),
        cmap=colmap,
        shading="gouraud",
        zorder=10,
        alpha=0.5,
    )

    plt.savefig(
        join(
            save_folder,
            "{:0>{width}d}_toy_class_gen.png".format(
                iteration, width=len(str(args["iterations"]))
            ),
        )
    )
    plt.close()
