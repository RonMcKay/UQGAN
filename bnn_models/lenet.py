from typing import Tuple

import bnn
import torch
import torch.nn as nn

from cls_models.base import BaseClassifier


class BLeNet(BaseClassifier):
    def __init__(
        self,
        cl_dim: int,
        image_channels: int = 1,
        image_size: int = 32,
        **kwargs,
    ) -> None:
        kwargs.pop("method", None)
        super().__init__(cl_dim=cl_dim, method="bayes", **kwargs)

        if image_size != 32:
            raise ValueError(
                (
                    "LeNet only works with images of size '32x32'! "
                    f"Got image size {image_size} instead!"
                )
            )

        self.cl_dim = cl_dim
        self.image_channels = image_channels
        self.image_size = image_size

        self.model = bnn.Sequential(
            bnn.BConv2d(in_channels=self.image_channels, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            bnn.BConv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            bnn.BLinear(in_features=16 * 5 * 5, out_features=120),
            nn.ReLU(),
            bnn.BLinear(in_features=120, out_features=84),
            nn.ReLU(),
            bnn.BLinear(in_features=84, out_features=self.cl_dim),
        )

        self.save_hyperparameters()

    def _forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self.model(x)
