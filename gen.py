from copy import deepcopy
from typing import Any, Dict

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.accelerators.registry import AcceleratorRegistry
import torch
from torch import Tensor, optim
import torch.nn as nn
import torch.nn.functional as tf

from cls_models.base import BaseClassifier, set_model_to_mode
from datasets import load_data
from eval_ood_detection import eval_classifier


class GEN(LightningModule):
    def __init__(
        self,
        classifier: BaseClassifier,
        generator: nn.Module,
        discriminator_image: nn.Module,
        discriminator_latent: nn.Module,
        vae: nn.Module,
        args: Dict,
        dataset: Dict,
        opt: Dict = {},
    ) -> None:
        super().__init__()

        self.classifier = classifier
        self.generator = generator
        self.discriminator_image = discriminator_image
        self.discriminator_latent = discriminator_latent
        self.vae = vae

        self.args = args
        self.dataset = dataset
        self.opt = opt

        self.save_hyperparameters(
            ignore=[
                "classifier",
                "generator",
                "discriminator_image",
                "discriminator_latent",
                "vae",
            ]
        )

        self.automatic_optimization = False

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        for module in (
            "classifier",
            "generator",
            "discriminator_image",
            "discriminator_latent",
            "vae",
        ):
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
                "weight_decay_cls", self.opt.get("weight_decay", 1e-3)
            ),
        )

        gen_opt = optim.RMSprop(
            self.generator.parameters(),
            lr=self.opt.get("lr_gen", self.opt.get("lr", 1e-4)),
        )

        disc_latent_opt = optim.RMSprop(
            self.discriminator_latent.parameters(),
            lr=self.opt.get("lr_disc_latent", self.opt.get("lr", 1e-4)),
            weight_decay=self.opt.get(
                "weight_decay_disc_latent", self.opt.get("weight_decay", 2e-4)
            ),
        )

        disc_image_opt = optim.RMSprop(
            self.discriminator_image.parameters(),
            lr=self.opt.get("lr_disc_image", self.opt.get("lr", 1e-4)),
            weight_decay=self.opt.get(
                "weight_decay_disc_image", self.opt.get("weight_decay", 2e-4)
            ),
        )

        vae_opt = optim.Adam(
            self.vae.parameters(), lr=self.opt.get("lr_vae", self.opt.get("lr", 1e-3))
        )

        return cls_opt, vae_opt, disc_latent_opt, disc_image_opt, gen_opt

    def reparameterize_code(self, code: Tensor, scale: Tensor) -> Tensor:
        code = code + torch.randn_like(code) * scale
        return code

    def kl_alpha(self, alpha: Tensor) -> Tensor:
        # Implemented as in https://muratsensoy.github.io/gen.html
        beta = torch.ones((1, alpha.shape[1]), device=alpha.device)
        s_alpha = alpha.sum(1, keepdim=True)
        s_beta = beta.sum()
        lnB = torch.lgamma(s_alpha) - torch.lgamma(alpha).sum(1, keepdim=True)
        lnB_uni = torch.lgamma(s_beta).neg()

        dg0 = torch.digamma(s_alpha)
        dg1 = torch.digamma(alpha)

        kl = ((alpha - beta) * (dg1 - dg0)).sum(1) + lnB + lnB_uni
        return kl.mean()

    def kl_vae(self, loc: Tensor, scale: Tensor, eps: float = 1e-8) -> Tensor:
        return (
            (loc**2 + scale**2 - torch.log(scale**2 + eps) - 1)
            .div(2)
            .sum(1)
            .mean()
        )

    def training_step(self, batch: Any, batch_idx: int) -> None:
        x, y = batch

        # Train Classifier
        self.generator.eval()
        self.discriminator_image.eval()
        self.discriminator_latent.eval()
        self.vae.eval()
        self.classifier.train()

        opt = self.optimizers()[0]
        opt.zero_grad()

        x_encoding, *_ = self.vae.encode(x)
        scale = self.generator(x_encoding)
        x_encoding_tilde = self.reparameterize_code(x_encoding, scale + 1e-3)
        x_tilde = torch.sigmoid(self.vae.decode(x_encoding_tilde))

        class_x = self.classifier._forward(x)
        class_x_tilde = self.classifier._forward(x_tilde)

        evidence = torch.exp(torch.clamp(class_x, max=80))  # to avoid overflows
        alpha = evidence + 1

        one_hot_classes = tf.one_hot(y.long(), num_classes=class_x.shape[1])

        class_loss_real = (
            tf.binary_cross_entropy_with_logits(
                class_x, one_hot_classes.float(), reduction="none"
            )[one_hot_classes == 1]
        ).mean()

        class_loss_fake = tf.binary_cross_entropy_with_logits(
            class_x_tilde,
            torch.zeros_like(class_x_tilde, dtype=torch.float, device=x.device),
            reduction="none",
        ).mean()

        class_loss = class_loss_real + class_loss_fake
        class_kl_loss = self.kl_alpha(
            alpha[one_hot_classes == 0].view(alpha.shape[0], -1)
        )

        beta = (
            (alpha / alpha.sum(1, keepdim=True))[one_hot_classes == 0]
            .reshape(alpha.shape[0], -1)
            .sum(1)
            .mean()
            .clone()
            .detach()
        )

        class_loss = class_loss + class_kl_loss * beta
        self.manual_backward(class_loss)

        self.log("class_loss", class_loss.item())
        self.log("class_loss_fake", class_loss_fake.item())
        self.log("class_loss_real", class_loss_real.item())
        self.log("class_kl_loss", class_kl_loss.item())
        self.log("beta", beta.item())
        opt.step()

        # Train VAE
        self.generator.eval()
        self.discriminator_image.eval()
        self.discriminator_latent.eval()
        self.vae.train()
        set_model_to_mode(self.classifier, "eval")

        opt = self.optimizers()[1]
        avg_vae_loss = 0
        for _ in range(self.args.get("vae_iterations", 1)):
            opt.zero_grad()

            x_encoding, x_reconstruction, loc_vae, scale_vae = self.vae(x)
            scale = self.generator(x_encoding)
            x_encoding_tilde = self.reparameterize_code(x_encoding, scale + 1e-3)

            disc_latent_tilde = self.discriminator_latent(x_encoding_tilde)

            vae_latent_loss = tf.binary_cross_entropy_with_logits(
                disc_latent_tilde,
                torch.zeros_like(disc_latent_tilde).to(torch.float),
            )

            kl = self.kl_vae(loc_vae, scale_vae)

            reconstruction_loss = (
                (torch.sigmoid(x_reconstruction) - x)
                .pow(2)
                .flatten(start_dim=1)
                .sum(1)
                .mean()
            )

            vae_loss = reconstruction_loss + vae_latent_loss + 0.1 * kl
            self.manual_backward(vae_loss)
            opt.step()
            avg_vae_loss += vae_loss.item()

        self.log(
            "vae_loss",
            avg_vae_loss / self.args.get("vae_iterations", 1),
        )

        # Train Latent Discriminator
        self.generator.eval()
        self.discriminator_latent.train()
        self.discriminator_image.eval()
        self.vae.eval()
        set_model_to_mode(self.classifier, "eval")

        opt = self.optimizers()[2]
        opt.zero_grad()

        x_encoding, *_ = self.vae.encode(x)
        scale = self.generator(x_encoding)
        x_encoding_tilde = self.reparameterize_code(x_encoding, scale + 1e-3)

        disc_latent_x = self.discriminator_latent(x_encoding)
        disc_latent_x_tilde = self.discriminator_latent(x_encoding_tilde)

        disc_latent_loss = tf.binary_cross_entropy_with_logits(
            disc_latent_x,
            torch.ones((x_encoding.shape[0],), dtype=torch.float, device=self.device),
        ) + tf.binary_cross_entropy_with_logits(
            disc_latent_x_tilde,
            torch.zeros(
                (x_encoding_tilde.shape[0],), dtype=torch.float, device=self.device
            ),
        )

        self.manual_backward(disc_latent_loss)

        self.log(
            "disc_latent_loss",
            disc_latent_loss.item(),
        )

        opt.step()

        # Train Image Discriminator
        self.generator.eval()
        self.discriminator_latent.eval()
        self.discriminator_image.train()
        self.vae.eval()
        set_model_to_mode(self.classifier, "eval")

        opt = self.optimizers()[3]
        opt.zero_grad()

        x_encoding, *_ = self.vae.encode(x)
        scale = self.generator(x_encoding)
        x_encoding_tilde = self.reparameterize_code(x_encoding, scale + 1e-3)
        x_tilde = torch.sigmoid(self.vae.decode(x_encoding_tilde))

        disc_img_x = self.discriminator_image(x)
        disc_img_x_tilde = self.discriminator_image(x_tilde)

        disc_img_loss_x = tf.binary_cross_entropy_with_logits(
            disc_img_x,
            torch.ones_like(disc_img_x).to(torch.float),
        )

        disc_img_loss_x_tilde = tf.binary_cross_entropy_with_logits(
            disc_img_x_tilde,
            torch.zeros_like(disc_img_x_tilde).to(torch.float),
        )

        disc_img_loss = disc_img_loss_x + disc_img_loss_x_tilde

        self.manual_backward(disc_img_loss)

        self.log("disc_img_loss", disc_img_loss.item())

        opt.step()

        # Train Generator
        self.generator.train()
        self.discriminator_image.eval()
        self.discriminator_latent.eval()
        self.vae.eval()
        set_model_to_mode(self.classifier, "eval")

        opt = self.optimizers()[4]
        opt.zero_grad()

        x_encoding, *_ = self.vae.encode(x)
        scale = self.generator(x_encoding)
        x_encoding_tilde = self.reparameterize_code(x_encoding, scale + 1e-3)
        x_tilde = torch.sigmoid(self.vae.decode(x_encoding_tilde))

        disc_latent_tilde = self.discriminator_latent(x_encoding_tilde)
        disc_img_tilde = self.discriminator_image(x_tilde)

        gen_disc_latent_loss = tf.binary_cross_entropy_with_logits(
            disc_latent_tilde,
            torch.ones(
                (disc_latent_tilde.shape[0],), dtype=torch.float, device=self.device
            ),
        )

        gen_disc_img_loss = tf.binary_cross_entropy_with_logits(
            disc_img_tilde,
            torch.zeros(
                (disc_img_tilde.shape[0],), dtype=torch.float, device=self.device
            ),
        )

        gen_loss = gen_disc_img_loss + gen_disc_latent_loss

        self.manual_backward(gen_loss)

        self.log("gen_loss", gen_loss.item())

        opt.step()

    def validation_step(self, batch, batch_idx, **kwargs) -> None:
        set_model_to_mode(self.classifier, "eval")
        x, y = batch
        y_hat = self.classifier(x)[0].argmax(1)
        val_acc = (y_hat == y).float().mean().item()
        self.log("val_acc", val_acc)

    def on_validation_epoch_end(self) -> None:
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

            accelerator = {
                v["accelerator"]: k for k, v in AcceleratorRegistry.items()
            }.get(type(self.trainer.accelerator))

            result = eval_classifier(
                classifier=deepcopy(self.classifier),
                trainer=Trainer(
                    logger=None,
                    enable_progress_bar=self.args.get("enable_progress_bar", False),
                    max_epochs=-1,
                    accelerator=accelerator,
                    devices=self.trainer.device_ids,
                ),
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
