import logging
from typing import Any, Dict, Optional, Tuple, Union

from bnn import KLLoss
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import Tensor, optim
import torch.nn.functional as tf
from torch.utils.data import DataLoader

from utils import entropy

from .cls_utils import set_model_to_mode

METHODS = ("softmax", "entropy", "uqgan", "mc-dropout", "gen", "bayes")
ClsOutput = Tuple[
    Tensor, Tensor, Tensor
]  # class probabilites, aleatoric uncertainty, epistemic uncertainty
BayesTrainOutput = Tuple[Tensor, Tensor]  # mc class probs and total KL Divergence


class BaseClassifier(LightningModule):
    def __init__(
        self,
        method: str,
        cl_dim: int,
        mc_samples: int = 1,
        *args: Dict[str, Any],
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"Unknown method '{method}'")
        self.method = method
        self.cl_dim = cl_dim
        self.mc_samples = mc_samples
        self.opt = kwargs.get("opt", {})
        self.average_mc_samples = kwargs.get("average_mc_samples", True)
        self.meta_classifier = kwargs.get("meta_classifier", False)

        if self.meta_classifier and self.method == "bayes":
            raise NotImplementedError(
                "Meta classifier with method 'bayes' is not implemented!"
            )

        self.console_logger = logging.getLogger("root.BaseClassifier")

        if self.method == "uqgan":
            self.register_buffer("_rel_class_frequencies", torch.ones((self.cl_dim,)))
            self.register_buffer("_neg_weight", torch.ones((self.cl_dim, self.cl_dim)))

    def compute_rel_class_frequencies(self, loader: DataLoader) -> None:
        self.console_logger.debug("Computing relative class frequencies")
        self._rel_class_frequencies = torch.zeros((self.cl_dim,))
        for _, y in loader:
            if self.meta_classifier and isinstance(y, (tuple, list)):
                y, _ = y
            self._rel_class_frequencies += tf.one_hot(
                y.long(), num_classes=self.cl_dim
            ).sum(0)

        self._neg_weight = self._rel_class_frequencies.unsqueeze(0).repeat(
            (self._rel_class_frequencies.shape[0], 1)
        ) * self._rel_class_frequencies.unsqueeze(0).repeat(
            (self._rel_class_frequencies.shape[0], 1)
        ).reciprocal().transpose(
            0, 1
        )

        self._rel_class_frequencies /= self._rel_class_frequencies.sum()
        self._rel_class_frequencies = self._rel_class_frequencies

    @property
    def rel_class_frequencies(self) -> Tensor:
        return self._rel_class_frequencies

    @property
    def neg_weight(self) -> Tensor:
        return self._neg_weight

    def compute_uqgan_prediction(
        self, x: Tensor, return_ova_probs: bool = False, *args, **kwargs
    ) -> ClsOutput:
        mc_probs = None
        mc_class_probs = None
        mc_in_dist_probs = None
        mc_samples = max(kwargs.get("mc_samples", self.mc_samples), 1)

        for _ in range(mc_samples):
            probs = torch.sigmoid(self._forward(x))
            if mc_probs is None:
                mc_probs = probs.unsqueeze(0)
                mc_class_probs, mc_in_dist_probs = [
                    out.unsqueeze(0)
                    for out in self.compute_uqgan_probs(
                        probs,
                    )
                ]
            else:
                mc_probs = torch.cat((mc_probs, probs.unsqueeze(0)))
                tmp_class_probs, tmp_in_dist_probs = self.compute_uqgan_probs(
                    probs,
                )
                mc_class_probs = torch.cat(
                    (mc_class_probs, tmp_class_probs.unsqueeze(0))
                )
                mc_in_dist_probs = torch.cat(
                    (mc_in_dist_probs, tmp_in_dist_probs.unsqueeze(0))
                )

        if self.average_mc_samples:
            mc_probs = mc_probs.mean(0)
            mc_class_probs = mc_class_probs.mean(0)
            mc_in_dist_probs = mc_in_dist_probs.mean(0)

        if return_ova_probs:
            return (
                mc_probs,
                entropy(mc_class_probs, dim=1 if self.average_mc_samples else 2),
                1 - mc_in_dist_probs,
            )
        else:
            return (
                mc_class_probs,
                entropy(mc_class_probs, dim=1 if self.average_mc_samples else 2),
                1 - mc_in_dist_probs,
            )

    def compute_uqgan_probs(
        self, probs: Tensor, eps: float = 1e-16
    ) -> Tuple[Tensor, Tensor]:
        probs = probs / ((2 - self.cl_dim) * probs + self.cl_dim - 1)
        probs = probs + eps
        class_probs = probs * self.rel_class_frequencies
        in_dist_probs = (probs.pow(2) * self.rel_class_frequencies).sum(1) / (
            probs * self.rel_class_frequencies
        ).sum(1)
        class_probs = class_probs / class_probs.sum(1, keepdim=True)
        return class_probs, in_dist_probs

    def compute_softmax_prediction(
        self, x: Tensor, use_entropy: bool = False, *args, **kwargs
    ) -> ClsOutput:
        mc_class_probs = None
        mc_samples = max(kwargs.get("mc_samples", self.mc_samples), 1)

        for _ in range(mc_samples):
            if mc_class_probs is None:
                mc_class_probs = torch.softmax(self._forward(x), dim=1).unsqueeze(0)
            else:
                mc_class_probs = torch.cat(
                    (
                        mc_class_probs,
                        torch.softmax(self._forward(x), dim=1).unsqueeze(0),
                    )
                )

        if self.average_mc_samples:
            mc_class_probs = mc_class_probs.mean(0)

        if use_entropy:
            uncert = entropy(mc_class_probs, dim=1 if self.average_mc_samples else 2)
        else:
            uncert = 1 - mc_class_probs.max(1 if self.average_mc_samples else 2)[0]

        return mc_class_probs, uncert, uncert

    def compute_gen_prediction(self, x: Tensor, *args, **kwargs) -> ClsOutput:
        out = self._forward(x)

        evidence = torch.exp(torch.clamp(out, max=80))  # to avoid overflows
        alpha = evidence + 1
        class_probs = alpha / alpha.sum(1, keepdim=True)
        uncert = entropy(class_probs)

        return class_probs, uncert, uncert

    def compute_bayes_prediction(
        self, x: Tensor, *args, **kwargs
    ) -> Union[ClsOutput, BayesTrainOutput]:
        mc_class_probs = torch.empty((0,), device=self.device)
        mc_samples = max(kwargs.get("mc_samples", self.mc_samples), 1)
        kl_total = torch.tensor(0.0, device=self.device, requires_grad=True)

        for _ in range(mc_samples):
            out, kl = self._forward(x)
            mc_class_probs = torch.cat((mc_class_probs, out.unsqueeze(0)))
            kl_total = kl_total + kl

        kl_total = kl_total / mc_samples

        if kwargs.get("mode", "eval") == "eval":
            mc_class_probs = torch.softmax(mc_class_probs, dim=2)
            au = entropy(mc_class_probs, dim=2).mean(0)
            mc_class_probs = mc_class_probs.mean(0)
            eu = entropy(mc_class_probs) - au

            return mc_class_probs, au, eu
        elif kwargs.get("mode", "eval") == "train":
            return mc_class_probs, kl_total

    def compute_prediction(
        self, x: Tensor, *args, **kwargs
    ) -> Union[ClsOutput, BayesTrainOutput]:
        if self.method == "uqgan":
            return self.compute_uqgan_prediction(x, *args, **kwargs)
        elif self.method == "softmax":
            return self.compute_softmax_prediction(x, *args, **kwargs)
        elif self.method == "entropy":
            return self.compute_softmax_prediction(x, use_entropy=True, *args, **kwargs)
        elif self.method == "mc-dropout":
            return self.compute_softmax_prediction(x, use_entropy=True, *args, **kwargs)
        elif self.method == "gen":
            return self.compute_gen_prediction(x, *args, **kwargs)
        elif self.method == "bayes":
            return self.compute_bayes_prediction(x, *args, **kwargs)

    def configure_optimizers(self):
        cls_opt = optim.Adam(
            self.parameters(),
            lr=self.opt.get("lr", 2e-4),
            weight_decay=self.opt.get("weight_decay", 2e-4),
            betas=(self.opt.get("beta1", 0.9), self.opt.get("beta2", 0.999)),
        )
        cls_scheduler = optim.lr_scheduler.LinearLR(
            cls_opt,
            start_factor=1.0,
            end_factor=self.opt.get("min_lr", 1e-5) / self.opt.get("lr", 2e-4),
            total_iters=self.trainer.max_epochs,
        )

        return [cls_opt], [cls_scheduler]

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Union[ClsOutput, BayesTrainOutput]:
        set_model_to_mode(self, "eval")
        x, _ = batch
        return self.compute_prediction(
            x,
            mc_samples=50
            if getattr(self, "mc_dropout", 0.0) > 0 or self.method == "bayes"
            else 1,
        )

    def training_step(self, batch, batch_idx, *args, **kwargs) -> Tensor:
        set_model_to_mode(self, "train")

        x, y = batch

        if self.meta_classifier and isinstance(y, (tuple, list)):
            y, h = y
        else:
            h = torch.ones_like(y)

        if self.method == "bayes":
            batch_weight = 2 ** (
                self.trainer.num_training_batches - (batch_idx + 1)
            ) / (2**self.trainer.num_training_batches - 1)

            outputs, kl_total = self.forward(x, mode="train")

            loss = KLLoss()(outputs, y, kl_total, batch_weight)
        else:
            loss = self.compute_loss(
                self.forward(x, return_ova_probs=True)[0],
                y,
                h,
            )

        self.log("train_loss", loss.item(), on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx, **kwargs) -> Optional[STEP_OUTPUT]:
        x, y = batch

        if self.meta_classifier and isinstance(y, (tuple, list)):
            y, h = y
            x = x[h == 1]
            y = y[h == 1]
            if x.shape[0] == 0:
                return

        # Use less mc_samples during validation than prediction to improve
        # training speed
        y_hat, *_ = self.compute_prediction(
            x,
            mc_samples=5
            if getattr(self, "mc_dropout", 0.0) > 0 or self.method == "bayes"
            else 1,
        )
        y_hat = y_hat.argmax(1)

        val_acc = (y == y_hat).float().mean().item()
        self.log("val_acc", val_acc)

    def forward(self, x: Tensor, *args, **kwargs) -> Union[ClsOutput, BayesTrainOutput]:
        return self.compute_prediction(x, *args, **kwargs)

    def compute_loss(
        self, out: torch.Tensor, y: torch.Tensor, h: torch.Tensor, eps: float = 1e-16
    ) -> Tensor:
        if self.method in ("softmax", "entropy", "mc-dropout"):
            # out is expected to be probabilities which is why we have to use the
            # logarithm and the nll_loss
            in_dist_loss = tf.nll_loss(torch.log(out[h == 1] + eps), y[h == 1].long())
            ood_loss = tf.kl_div(
                torch.log(out[h == 0] + eps),
                torch.full_like(
                    out[h == 0],
                    1.0 / out.shape[1],
                    dtype=torch.float,
                    device=out.device,
                ),
            )
        elif self.method == "uqgan":
            in_dist_mask = tf.one_hot(
                y[h == 1].long(), num_classes=out.shape[1]
            )  # type: torch.Tensor
            class_loss = tf.binary_cross_entropy(
                out[h == 1], in_dist_mask.float(), reduction="none"
            )
            in_dist_loss_one = (
                class_loss[in_dist_mask == 1].sum().div(in_dist_mask.shape[0])
            )
            in_dist_loss_all = (
                class_loss[in_dist_mask == 0]
                * self.neg_weight.index_select(0, y[h == 1].long())[in_dist_mask == 0]
            ).mean()
            ood_loss = tf.binary_cross_entropy(
                out[h == 0],
                torch.zeros_like(out[h == 0], dtype=torch.float),
                reduction="mean",
            )
            return in_dist_loss_one + in_dist_loss_all * 0.5 + ood_loss * 0.5
        else:
            raise ValueError(f"Unknown method '{self.method}' for loss calculation.")

        if torch.isnan(ood_loss):
            if not globals().get("_issued_ood_loss_nan_warning", False):
                global _issued_ood_loss_nan_warning
                _issued_ood_loss_nan_warning = True
                self.console_logger.warning(
                    "'ood_loss' is nan! Only returning 'in_dist_loss'."
                )

            return in_dist_loss
        else:
            return (in_dist_loss + ood_loss).div(2.0)
