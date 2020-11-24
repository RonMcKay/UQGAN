from typing import Any, Dict, List, Tuple

from pytorch_lightning import LightningModule
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as tf
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from utils import save_sample_images


class BaseCAE(LightningModule):
    def __init__(
        self,
        num_sample_images: int = 8,
        sample_every_n: int = 5,
        *args: Tuple[Any],
        **kwargs: Dict[str, Any]
    ) -> None:
        super().__init__()

        self.num_sample_images = num_sample_images
        self.sample_every_n = sample_every_n

        self.opt = kwargs.get("opt", {})

        self.reconstruction_crit = nn.BCEWithLogitsLoss()

        self.save_hyperparameters()

    def configure_optimizers(self):
        cae_opt = Adam(
            self.parameters(),
            lr=self.opt.get("lr", 1e-3),
            weight_decay=self.opt.get("weight_decay", 0),
        )
        cae_scheduler = MultiStepLR(cae_opt, milestones=[50, 100], gamma=0.1)

        return [cae_opt], [cae_scheduler]

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        x, y = batch
        y_oh = tf.one_hot(y, self.cl_dim)

        reconstruction, _ = self.forward(x, y_oh)

        loss = self.reconstruction_crit(reconstruction, x)
        self.log("train_loss", loss.item(), on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Tuple[Tensor, Tensor]:
        x: Tensor
        y: Tensor
        x, y = batch
        y_oh = tf.one_hot(y, num_classes=self.cl_dim)

        reconstruction, _ = self.forward(x, y_oh)
        loss = self.reconstruction_crit(reconstruction, x)

        self.log("val_loss", loss.item(), on_epoch=True, on_step=False)

        return x, torch.sigmoid(reconstruction)

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tuple[Tensor, Tensor]:
        x: Tensor
        y: Tensor
        x, y = batch
        y_oh = tf.one_hot(y, num_classes=self.cl_dim)

        reconstruction, embedding = self.forward(x, y_oh)

        return torch.sigmoid(reconstruction), embedding

    def validation_epoch_end(self, outputs: List[Tuple[Tensor, Tensor]]) -> None:
        if (self.trainer.current_epoch + 1) % self.sample_every_n == 0:
            originals = torch.cat([i[0] for i in outputs])
            reconstructions = torch.cat([i[1] for i in outputs])

            if not hasattr(self, "sample_indices"):
                self.sample_indices = torch.randperm(originals.shape[0])[
                    : self.num_sample_images
                ]
            originals = originals[self.sample_indices]
            reconstructions = reconstructions[self.sample_indices]

            save_sample_images(
                folder=self.trainer.default_root_dir,
                images=reconstructions.cpu(),
                iteration=self.trainer.current_epoch + 1,
                original_images=originals.cpu(),
            )

    def forward(self, x: Tensor, cl: Tensor) -> Tuple[Tensor, Tensor]:
        encoding = self.encode(x, cl)
        reconstruction = self.decode(encoding, cl)

        return reconstruction, encoding

    def encode(self, x: Tensor, cl: Tensor) -> Tensor:
        return self._encode(x, cl)

    def decode(self, encoding: Tensor, cl: Tensor) -> Tensor:
        return self._decode(encoding, cl)

    def _encode(self, x: Tensor, cl: Tensor) -> Tensor:
        raise NotImplementedError

    def _decode(self, encoding: Tensor, cl: Tensor) -> Tensor:
        raise NotImplementedError
