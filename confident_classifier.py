from copy import deepcopy
import logging
from typing import Any, Dict

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as tf

from cls_models import set_model_to_mode
from cls_models.base import BaseClassifier
from datasets import load_data
from eval_ood_detection import eval_classifier


class ConfidentClassifier(LightningModule):
    def __init__(
        self,
        classifier: BaseClassifier,
        generator: nn.Module,
        discriminator: nn.Module,
        args: Dict,
        dataset: Dict,
        opt: Dict = {},
    ) -> None:
        super().__init__()

        self.classifier = classifier
        self.generator = generator
        self.discriminator = discriminator

        self.args = args
        self.dataset = dataset
        self.opt = opt

        self.save_hyperparameters(ignore=["classifier", "generator", "discriminator"])

        self.class_crit = nn.NLLLoss()
        self.kl_crit = nn.KLDivLoss()
        self.gan_crit = nn.BCEWithLogitsLoss()

        self.console_logger = logging.getLogger("root.ConfidentClassifier")

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        for module in ("classifier", "generator", "discriminator"):
            if hasattr(getattr(self, module), "_hparams_name"):
                checkpoint[f"{module}_hparams_name"] = getattr(
                    self, module
                )._hparams_name
                checkpoint[f"{module}_hyper_parameters"] = getattr(self, module).hparams

    def configure_optimizers(self):
        disc_opt = optim.Adam(
            self.discriminator.parameters(),
            lr=self.opt.get("lr_disc", self.opt.get("lr", 2e-4)),
            weight_decay=self.opt.get(
                "weight_decay_disc", self.opt.get("weight_decay", 2e-4)
            ),
            betas=(0, 0.9),
        )
        disc_sched = optim.lr_scheduler.LinearLR(
            disc_opt,
            start_factor=1.0,
            end_factor=self.opt.get("min_lr_disc", self.opt.get("min_lr", 1e-5))
            / self.opt.get("lr_disc", self.opt.get("lr", 2e-4)),
            total_iters=self.trainer.estimated_stepping_batches,
        )

        cls_opt = optim.Adam(
            self.classifier.parameters(),
            lr=self.opt.get("lr_cls", self.opt.get("lr", 2e-4)),
            weight_decay=self.opt.get(
                "weight_decay_cls", self.opt.get("weight_decay", 2e-4)
            ),
            betas=(0, 0.9),
        )
        cls_sched = optim.lr_scheduler.LinearLR(
            cls_opt,
            start_factor=1.0,
            end_factor=self.opt.get("min_lr_cls", self.opt.get("min_lr", 1e-5))
            / self.opt.get("lr_cls", self.opt.get("lr", 2e-4)),
            total_iters=self.trainer.estimated_stepping_batches,
        )

        gen_opt = optim.Adam(
            self.generator.parameters(),
            lr=self.opt.get("lr_gen", self.opt.get("lr", 2e-4)),
            weight_decay=self.opt.get(
                "weight_decay_gen", self.opt.get("weight_decay", 2e-4)
            ),
            betas=(0, 0.9),
        )
        gen_sched = optim.lr_scheduler.LinearLR(
            gen_opt,
            start_factor=1.0,
            end_factor=self.opt.get("min_lr_gen", self.opt.get("min_lr", 1e-5))
            / self.opt.get("lr_gen", self.opt.get("lr", 2e-4)),
            total_iters=self.trainer.estimated_stepping_batches,
        )

        return [disc_opt, gen_opt, cls_opt], [disc_sched, gen_sched, cls_sched]

    def training_step(self, batch, batch_idx, optimizer_idx) -> STEP_OUTPUT:
        x, y = batch

        if optimizer_idx == 0:
            ############################
            #    Train Discriminator
            ############################
            self.generator.eval()
            set_model_to_mode(self.classifier, "eval")

            with torch.no_grad():
                x_tilde = self.generator(num_samples=x.shape[0])
                if "image_channels" in self.dataset["cfg"]:
                    x_tilde = torch.sigmoid(x_tilde)

            disc_x_tilde = self.discriminator(x_tilde)
            disc_x = self.discriminator(x)

            disc_loss = self.gan_crit(
                disc_x_tilde, torch.zeros_like(disc_x_tilde)
            ) + self.gan_crit(
                disc_x, torch.ones_like(disc_x)
            )  # type: torch.Tensor

            self.log("disc_loss", disc_loss.item(), on_epoch=True, on_step=False)

            return disc_loss
        elif optimizer_idx == 1:
            ############################
            #    Train Generator
            ############################
            self.discriminator.eval()
            set_model_to_mode(self.classifier, "eval")

            x_tilde = self.generator(num_samples=x.shape[0])
            if "image_channels" in self.dataset["cfg"]:
                x_tilde = torch.sigmoid(x_tilde)

            disc_x_tilde = self.discriminator(x_tilde)

            class_x_tilde = self.classifier(x_tilde)[0]
            class_loss_x_tilde = self.kl_crit(
                tf.log_softmax(class_x_tilde, dim=1),
                torch.full_like(class_x_tilde, 1.0 / class_x_tilde.shape[1]),
            )

            gen_loss = (
                -self.gan_crit(disc_x_tilde, torch.zeros_like(disc_x_tilde))
                + self.args["beta"] * class_loss_x_tilde
            )

            self.log("gen_loss", gen_loss.item(), on_epoch=True, on_step=False)

            return gen_loss
        elif optimizer_idx == 2:
            ############################
            #    Train Classifier
            ############################
            self.generator.eval()
            self.discriminator.eval()

            with torch.no_grad():
                x_tilde = self.generator(num_samples=x.shape[0])
                if "image_channels" in self.dataset["cfg"]:
                    x_tilde = torch.sigmoid(x_tilde)

            class_x = self.classifier(x)[0]
            class_x_tilde = self.classifier(x_tilde)[0]

            class_loss_x_tilde = self.kl_crit(
                tf.log_softmax(class_x_tilde, dim=1),
                torch.full_like(class_x_tilde, 1.0 / class_x_tilde.shape[1]),
            )
            class_loss_x = self.class_crit(torch.log(class_x + 1e-16), y)

            class_loss = class_loss_x + self.args["beta"] * class_loss_x_tilde

            self.log("class_loss", class_loss.item(), on_epoch=True, on_step=False)

            return class_loss

    def validation_step(self, batch, batch_idx, **kwargs) -> None:
        set_model_to_mode(self.classifier, "eval")
        x, y = batch
        y_hat = self.classifier(x)[0].argmax(1)
        val_acc = (y_hat == y).float().mean().item()
        self.log("val_acc", val_acc, on_epoch=True)

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
