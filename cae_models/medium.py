from typing import Any

import torch
from torch import Tensor
import torch.nn as nn

from .base import BaseCAE


class MediumCAE(BaseCAE):
    def __init__(
        self,
        cl_dim: int,
        image_channels: int = 1,
        image_size: int = 28,
        num_channels: int = 64,
        latent_dim: int = 128,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cl_dim = cl_dim

        self.encoder = Encoder(
            cl_dim=self.cl_dim,
            image_channels=image_channels,
            image_size=image_size,
            num_channels=num_channels,
            latent_dim=latent_dim,
        )

        self.decoder = Decoder(
            cl_dim=self.cl_dim,
            image_channels=image_channels,
            image_size=image_size,
            num_channels=num_channels,
            latent_dim=latent_dim,
        )

    def _encode(self, x: Tensor, cl: Tensor) -> Tensor:
        return self.encoder(x, cl)

    def _decode(self, encoding: Tensor, cl: Tensor) -> Tensor:
        return self.decoder(encoding, cl)


class Encoder(nn.Module):
    def __init__(
        self, cl_dim, image_channels=1, image_size=32, num_channels=64, latent_dim=128
    ):
        super().__init__()

        if (image_size / 8.0) % 1 != 0:
            raise ValueError("'image_size' should be divisible by 8.")

        self.cl_dim = cl_dim
        self.image_channels = image_channels
        self.image_size = image_size
        self.num_channels = num_channels
        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            # Downsample 1
            self.conv_block_down(
                in_channels=self.image_channels + self.cl_dim,
                out_channels=self.num_channels // 4,
            ),
            # Downsample 2
            self.conv_block_down(
                in_channels=self.num_channels // 4, out_channels=self.num_channels // 2
            ),
            # Downsample 3
            self.conv_block_down(
                in_channels=self.num_channels // 2, out_channels=self.num_channels
            ),
            # Embedding output
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(
                in_features=self.num_channels * (self.image_size // 8) ** 2,
                out_features=self.latent_dim,
            ),
        )

    def forward(self, x, cl):
        cl = cl.unsqueeze(-1).unsqueeze(-1)
        cl = cl.expand(list(cl.shape)[:2] + [x.shape[2], x.shape[3]])
        x = torch.cat((x, cl), 1)

        out = self.model(x)

        encoding = self.fc(out)

        return encoding

    def conv_block_down(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
        return block


class Decoder(nn.Module):
    def __init__(
        self, cl_dim, image_channels=1, image_size=28, num_channels=64, latent_dim=128
    ):
        super().__init__()

        if (image_size / 8.0) % 1 != 0:
            raise ValueError("'image_size' should be divisible by 8.")

        self.cl_dim = cl_dim
        self.image_channels = image_channels
        self.image_size = image_size
        self.num_channels = num_channels
        self.latent_dim = latent_dim

        self.fc = nn.Sequential(
            nn.Linear(
                in_features=latent_dim + cl_dim,
                out_features=self.num_channels * (self.image_size // 8) ** 2,
            ),
            nn.ReLU(),
        )

        self.model = nn.Sequential(
            # Upsample 1
            self.conv_block_up(
                in_channels=self.num_channels, out_channels=self.num_channels // 2
            ),
            # Upsample 2
            self.conv_block_up(
                in_channels=self.num_channels // 2, out_channels=self.num_channels // 4
            ),
            # Upsample 3
            self.conv_block_up(
                in_channels=self.num_channels // 4, out_channels=self.num_channels // 4
            ),
            nn.Conv2d(
                in_channels=self.num_channels // 4,
                out_channels=self.image_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, encoding, cl):
        out = torch.cat((encoding, cl), dim=1)
        out = self.fc(out)
        out = out.reshape(
            -1, self.num_channels, self.image_size // 8, self.image_size // 8
        )

        out = self.model(out)
        return out

    def conv_block_up(self, in_channels, out_channels, act=nn.LeakyReLU(0.2)):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            act,
        )
