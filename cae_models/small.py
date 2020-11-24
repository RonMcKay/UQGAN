from typing import Any, Dict, Tuple

import torch
from torch import Tensor
import torch.nn as nn

from .base import BaseCAE


class SmallCAE(BaseCAE):
    def __init__(
        self,
        cl_dim: int,
        image_channels: int = 1,
        image_size: int = 28,
        latent_dim: int = 8,
        *args: Tuple[Any],
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cl_dim = cl_dim

        self.encoder = Encoder(
            cl_dim=self.cl_dim,
            image_channels=image_channels,
            image_size=image_size,
            latent_dim=latent_dim,
        )

        self.decoder = Decoder(
            cl_dim=self.cl_dim,
            image_channels=image_channels,
            image_size=image_size,
            latent_dim=latent_dim,
        )

    def _encode(self, x: Tensor, cl: Tensor) -> Tensor:
        return self.encoder(x, cl)

    def _decode(self, encoding: Tensor, cl: Tensor) -> Tensor:
        return self.decoder(encoding, cl)


class Encoder(nn.Module):
    def __init__(
        self,
        cl_dim: int,
        image_channels: int = 1,
        image_size: int = 28,
        latent_dim: int = 8,
    ) -> None:
        super().__init__()

        if (image_size / 4.0) % 1 != 0:
            raise ValueError("'image_size' should be divisible by 4.")

        self.cl_dim = cl_dim
        self.image_channels = image_channels
        self.image_size = image_size
        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=self.image_channels + self.cl_dim,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.fc = nn.Linear(
            in_features=64 * (self.image_size // 4) ** 2, out_features=self.latent_dim
        )

    def forward(self, x: Tensor, cl: Tensor) -> Tensor:
        cl = cl.unsqueeze(-1).unsqueeze(-1)
        cl = cl.expand(list(cl.shape)[:2] + [x.shape[2], x.shape[3]])
        x = torch.cat((x, cl), 1)

        out = self.model(x)

        encoding = self.fc(out)

        return encoding


class Decoder(nn.Module):
    def __init__(
        self,
        cl_dim: int,
        image_channels: int = 1,
        image_size: int = 28,
        latent_dim: int = 8,
    ) -> None:
        super().__init__()

        if (image_size / 4.0) % 1 != 0:
            raise ValueError("'image_size' should be divisible by 4.")

        self.cl_dim = cl_dim
        self.image_channels = image_channels
        self.image_size = image_size
        self.latent_dim = latent_dim

        self.fc = nn.Sequential(
            nn.Linear(
                in_features=latent_dim + cl_dim,
                out_features=64 * (self.image_size // 4) ** 2,
            ),
            nn.ReLU(),
        )

        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=self.image_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                output_padding=0,
            ),
        )

    def forward(self, encoding: Tensor, cl: Tensor) -> Tensor:
        out = torch.cat((encoding, cl), dim=1)
        out = self.fc(out)
        out = out.reshape(-1, 64, self.image_size // 4, self.image_size // 4)

        out = self.model(out)
        return out
