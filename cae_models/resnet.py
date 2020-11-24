# This is a PyTorch reimplementation of the code to the paper
# 'Improved Training of Wasserstein GANs'
# https://github.com/igul222/improved_wgan_training

import logging
from math import ceil, floor, log2
from typing import Any, Dict, Tuple

import torch
from torch import Tensor
import torch.nn as nn

from utils import init_weights

from .base import BaseCAE

LATENT_DIM = 128
N_FEATUREMAPS = 32


class SubpixelConv2D(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels * 4,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.model(x)


class ConvMeanPool(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, bias=True
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.AvgPool2d(2),
        )

    def forward(self, x):
        return self.model(x)


class MeanPoolConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, bias=True
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
        )

    def forward(self, x):
        return self.model(x)


class UpsampleConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, bias=True
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
        )

    def forward(self, x):
        return self.model(x)


class BottleneckResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        resample=None,
        norm=None,
        input_spatial_size=None,
        act: nn.Module = nn.ReLU(),
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act

        if resample == "down":
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2,
                padding=0,
            )
            self.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
            # compute padding for given kernel size so that spatial
            # size halves with stride of 2
            # this assumes that image size is a power of 2
            padding = floor(kernel_size / 2.0)
            self.conv1b = nn.Conv2d(
                in_channels=in_channels // 2,
                out_channels=out_channels // 2,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                bias=True,
            )
            self.conv2 = nn.Conv2d(
                in_channels=out_channels // 2,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
        elif resample == "up":
            self.shortcut = SubpixelConv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
            self.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
            # compute padding and output padding for given kernel size so that
            # spatial size doubles with stride of 2
            padding = ceil(kernel_size / 2.0) - 1
            output_padding = 2 + 2 * padding - kernel_size
            self.conv1b = nn.ConvTranspose2d(
                in_channels=in_channels // 2,
                out_channels=out_channels // 2,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=True,
            )
            self.conv2 = nn.Conv2d(
                in_channels=out_channels // 2,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
        elif resample is None:
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
            self.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
            self.conv1b = nn.Conv2d(
                in_channels=in_channels // 2,
                out_channels=out_channels // 2,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
            )
            self.conv2 = nn.Conv2d(
                in_channels=out_channels // 2,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
        else:
            raise ValueError(
                (
                    f"Unknown resample type '{resample}'. "
                    "Valid options are: [None, 'down', 'up']"
                )
            )

        if in_channels == out_channels and resample is None:
            self.shortcut = nn.Identity()

        if norm == "batchnorm":
            self.norm = nn.BatchNorm2d(num_features=out_channels)
        elif norm == "layernorm":
            if input_spatial_size is None:
                raise ValueError(
                    "Please specify 'input_spatial_size' if norm='layernorm'."
                )
            if resample == "up":
                self.norm = nn.LayerNorm(
                    [out_channels] + [2 * d for d in input_spatial_size]
                )
            elif resample == "down":
                self.norm = nn.LayerNorm(
                    [out_channels] + [d // 2 for d in input_spatial_size]
                )
            elif resample is None:
                self.norm = nn.LayerNorm([out_channels] + input_spatial_size)
        elif norm is None:
            self.norm = nn.Identity()
        else:
            raise ValueError(
                (
                    f"Unknown normalization type '{norm}'. "
                    "Valid options are: [None, 'batchnorm', 'layernorm']"
                )
            )

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.act(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv1b(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm(x)
        return shortcut + 0.3 * x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        resample=None,
        norm=None,
        input_spatial_size=None,
        act: nn.Module = nn.ReLU(),
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act

        if resample == "down":
            self.shortcut = MeanPoolConv(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )
            self.conv1 = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            self.conv2 = ConvMeanPool(
                in_channels, out_channels, kernel_size, stride, padding
            )
        elif resample == "up":
            self.shortcut = UpsampleConv(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )
            self.conv1 = UpsampleConv(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            self.conv2 = nn.Conv2d(
                out_channels, out_channels, kernel_size, stride=stride, padding=padding
            )
        elif resample is None:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )
            self.conv1 = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            self.conv2 = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding
            )
        else:
            raise ValueError(
                (
                    f"Unknown resample type '{resample}'. "
                    "Valid options are: [None, 'down', 'up']"
                )
            )

        if in_channels == out_channels and resample is None:
            self.shortcut = nn.Identity()

        if norm == "batchnorm":
            self.norm1 = nn.BatchNorm2d(num_features=in_channels)
            self.norm2 = nn.BatchNorm2d(num_features=out_channels)
        elif norm == "layernorm":
            if input_spatial_size is None:
                raise ValueError(
                    "Please specify 'input_spatial_size' if norm='layernorm'."
                )
            self.norm1 = nn.LayerNorm([in_channels] + input_spatial_size)
            if resample == "up":
                self.norm2 = nn.LayerNorm(
                    [out_channels] + [d * 2 for d in input_spatial_size]
                )
            else:
                self.norm2 = nn.LayerNorm([in_channels] + input_spatial_size)
        elif norm is None:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        else:
            raise ValueError("Invalid normalization mode")

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        return shortcut + x


class ResNetCAE(BaseCAE):
    def __init__(
        self,
        cl_dim: int,
        image_channels: int = 3,
        image_size: int = 32,
        n_featuremaps: int = N_FEATUREMAPS,
        latent_dim: int = LATENT_DIM,
        *args: Tuple[Any],
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cl_dim = cl_dim

        self.encoder = Encoder(
            cl_dim=self.cl_dim,
            n_featuremaps=n_featuremaps,
            latent_dim=latent_dim,
            image_channels=image_channels,
            image_size=image_size,
            **kwargs,
        )

        self.decoder = Decoder(
            cl_dim=self.cl_dim,
            n_featuremaps=n_featuremaps,
            latent_dim=latent_dim,
            image_channels=image_channels,
            image_size=image_size,
            **kwargs,
        )

    def _encode(self, x: Tensor, cl: Tensor) -> Tensor:
        return self.encoder(x, cl)

    def _decode(self, encoding: Tensor, cl: Tensor) -> Tensor:
        return self.decoder(encoding, cl)


class Decoder(nn.Module):
    def __init__(
        self,
        cl_dim,
        n_featuremaps=N_FEATUREMAPS,
        blocks=1,
        latent_dim=LATENT_DIM,
        norm="batchnorm",
        image_channels=3,
        image_size=64,
        max_featuremaps=512,
        **kwargs,
    ):
        super().__init__()

        if log2(image_size) % 1 != 0:
            raise ValueError("'img_size' has to be a power of 2.")

        if log2(n_featuremaps) % 1 != 0:
            raise ValueError("'n_featuremaps' has to be a power of 2.")

        self.n_featuremaps = n_featuremaps
        self.latent_dim = latent_dim
        self.img_size = image_size
        self.max_featuremaps = max_featuremaps
        self.act = nn.ReLU()
        self.n_up = floor(log2(self.img_size // 4))

        if isinstance(blocks, (list, tuple)) and len(blocks) != self.n_up:
            raise ValueError(
                "If blocks is a list of integers it should be one integer per "
                f"up sampling step ({self.n_up})"
            )
        elif isinstance(blocks, int):
            self.blocks = tuple([blocks] * self.n_up)
        elif isinstance(blocks, list):
            self.blocks = tuple(blocks)

        self.fc = nn.Linear(
            cl_dim + self.latent_dim,
            4**2 * min(2 ** (self.n_up - 1) * self.n_featuremaps, max_featuremaps),
        )

        self.model = nn.Sequential()
        self.model.add_module(
            "Decoder_4x4" + ("_0" if self.blocks[0] > 1 else ""),
            ResidualBlock(
                in_channels=min(
                    2 ** (self.n_up - 1) * self.n_featuremaps, max_featuremaps
                ),
                out_channels=min(
                    2 ** (self.n_up - 1) * self.n_featuremaps, max_featuremaps
                ),
                resample="up",
                norm=norm,
                act=self.act,
            ),
        )
        for j in range(self.blocks[0] - 1):
            self.model.add_module(
                "Decoder_4x4_{0}".format(j + 1),
                ResidualBlock(
                    in_channels=min(
                        2 ** (self.n_up - 1) * self.n_featuremaps, max_featuremaps
                    ),
                    out_channels=min(
                        2 ** (self.n_up - 1) * self.n_featuremaps, max_featuremaps
                    ),
                    norm=norm,
                    act=self.act,
                ),
            )

        for i in range(1, self.n_up):
            scale_factor = 2 ** (self.n_up - i)
            self.model.add_module(
                "Decoder_{0}x{0}".format(self.img_size // scale_factor)
                + ("_0" if self.blocks[i] > 1 else ""),
                ResidualBlock(
                    in_channels=min(scale_factor * self.n_featuremaps, max_featuremaps),
                    out_channels=min(
                        scale_factor // 2 * self.n_featuremaps, max_featuremaps
                    ),
                    resample="up",
                    norm=norm,
                    act=self.act,
                ),
            )
            for j in range(self.blocks[i] - 1):
                self.model.add_module(
                    "Decoder_{0}x{0}_{1}".format(self.img_size // scale_factor, j + 1),
                    ResidualBlock(
                        in_channels=min(
                            scale_factor // 2 * self.n_featuremaps, max_featuremaps
                        ),
                        out_channels=min(
                            scale_factor // 2 * self.n_featuremaps, max_featuremaps
                        ),
                        norm=norm,
                        act=self.act,
                    ),
                )

        self.model.add_module(
            "Decoder_Norm_Out",
            nn.BatchNorm2d(num_features=self.n_featuremaps),
        )
        self.model.add_module(
            "Decoder_Act_Out",
            self.act,
        )
        self.model.add_module(
            "Decoder_Conv_Out",
            nn.Conv2d(
                in_channels=self.n_featuremaps,
                out_channels=image_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        self.apply(init_weights)

    def forward(self, encoding, cl):
        out = self.fc(torch.cat((encoding, cl), dim=1))
        out = self.model(
            out.reshape(
                cl.shape[0],
                min(2 ** (self.n_up - 1) * self.n_featuremaps, self.max_featuremaps),
                4,
                4,
            )
        )
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        cl_dim,
        n_featuremaps=N_FEATUREMAPS,
        blocks=1,
        norm="layernorm",
        latent_dim=LATENT_DIM,
        image_channels=3,
        image_size=64,
        max_featuremaps=512,
        **kwargs,
    ):
        super().__init__()

        if log2(image_size) % 1 != 0:
            raise ValueError("'img_size' has to be a power of 2.")

        if log2(n_featuremaps) % 1 != 0:
            raise ValueError("'n_featuremaps' has to be a power of 2.")

        self.n_featuremaps = n_featuremaps
        self.img_size = image_size
        self.act = nn.LeakyReLU()
        self.n_down = floor(log2(self.img_size // 4))

        if isinstance(blocks, (list, tuple)) and len(blocks) != self.n_down:
            raise ValueError(
                "If blocks is a list of integers it should be one integer per "
                f"down sampling step ({self.n_down})"
            )
        elif isinstance(blocks, int):
            self.blocks = tuple([blocks] * self.n_down)
        elif isinstance(blocks, list):
            self.blocks = tuple(blocks)

        self.model = nn.Sequential()
        self.model.add_module(
            "Encoder_In",
            nn.Conv2d(
                in_channels=image_channels + cl_dim,
                out_channels=self.n_featuremaps,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        for i in range(self.n_down - 1):
            scale_factor = 2**i
            self.model.add_module(
                "Encoder_{0}x{0}".format(self.img_size // scale_factor)
                + ("_0" if self.blocks[i] > 1 else ""),
                ResidualBlock(
                    in_channels=min(scale_factor * self.n_featuremaps, max_featuremaps),
                    out_channels=min(
                        2 * scale_factor * self.n_featuremaps, max_featuremaps
                    ),
                    resample="down",
                    norm=norm,
                    act=self.act,
                    input_spatial_size=[self.img_size // scale_factor] * 2,
                ),
            )
            for j in range(self.blocks[i] - 1):
                self.model.add_module(
                    "Encoder_{0}x{0}_{1}".format(self.img_size // scale_factor, j + 1),
                    ResidualBlock(
                        in_channels=min(
                            2 * scale_factor * self.n_featuremaps, max_featuremaps
                        ),
                        out_channels=min(
                            2 * scale_factor * self.n_featuremaps, max_featuremaps
                        ),
                        norm=norm,
                        act=self.act,
                        input_spatial_size=[self.img_size // scale_factor // 2] * 2,
                    ),
                )

        scale_factor = 2 ** (self.n_down - 1)
        self.model.add_module(
            "Encoder_{0}x{0}".format(self.img_size // scale_factor)
            + ("_0" if self.blocks[-1] > 1 else ""),
            ResidualBlock(
                in_channels=min(scale_factor * self.n_featuremaps, max_featuremaps),
                out_channels=min(scale_factor * self.n_featuremaps, max_featuremaps),
                resample="down",
                norm=norm,
                act=self.act,
                input_spatial_size=[self.img_size // scale_factor] * 2,
            ),
        )
        for j in range(self.blocks[-1] - 1):
            self.model.add_module(
                "Encoder_{0}x{0}_{1}".format(self.img_size // scale_factor, j + 1),
                ResidualBlock(
                    in_channels=min(scale_factor * self.n_featuremaps, max_featuremaps),
                    out_channels=min(
                        scale_factor * self.n_featuremaps, max_featuremaps
                    ),
                    norm=norm,
                    act=self.act,
                    input_spatial_size=[self.img_size // scale_factor // 2] * 2,
                ),
            )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=4**2
                * min(2 ** (self.n_down - 1) * self.n_featuremaps, max_featuremaps),
                out_features=latent_dim,
            ),
        )

        self.apply(init_weights)

    def forward(self, x, cl):
        cl = cl.unsqueeze(-1).unsqueeze(-1)
        cl = cl.expand(list(cl.shape)[:2] + [x.shape[2], x.shape[3]])
        x = torch.cat((x, cl), 1)

        x = self.model(x)
        encoding = self.fc(x)

        return encoding


class DeepResNetCAE(BaseCAE):
    def __init__(
        self,
        cl_dim: int,
        image_channels: int = 3,
        image_size: int = 32,
        n_featuremaps: int = N_FEATUREMAPS,
        latent_dim: int = LATENT_DIM,
        *args: Tuple[Any],
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = DeepEncoder(
            cl_dim=cl_dim,
            n_featuremaps=n_featuremaps,
            latent_dim=latent_dim,
            image_channels=image_channels,
            image_size=image_size,
            **kwargs,
        )

        self.decoder = DeepDecoder(
            cl_dim=cl_dim,
            n_featuremaps=n_featuremaps,
            latent_dim=latent_dim,
            image_channels=image_channels,
            image_size=image_size,
            **kwargs,
        )

    def _encode(self, x: Tensor, cl: Tensor) -> Tensor:
        return self.encoder(x, cl)

    def _decode(self, encoding: Tensor, cl: Tensor) -> Tensor:
        return self.decoder(encoding, cl)


class DeepDecoder(nn.Module):
    def __init__(
        self,
        cl_dim,
        n_featuremaps=N_FEATUREMAPS,
        latent_dim=LATENT_DIM,
        norm="batchnorm",
        image_channels=3,
        image_size=64,
        max_featuremaps=512,
        n_layers=6,
        **kwargs,
    ):
        super().__init__()

        if log2(image_size) % 1 != 0:
            raise ValueError("'img_size' has to be a power of 2.")

        if log2(n_featuremaps) % 1 != 0:
            raise ValueError("'n_featuremaps' has to be a power of 2.")

        if not n_layers > 1:
            raise ValueError("'n_layers' should be greater than 1.")

        self.n_featuremaps = n_featuremaps
        self.latent_dim = latent_dim
        self.img_size = image_size
        self.max_featuremaps = max_featuremaps
        self.n_up = floor(log2(self.img_size // 4))
        self.act = nn.ReLU()

        self.fc = nn.Linear(
            cl_dim + self.latent_dim,
            4**2 * min(2 ** (self.n_up - 1) * self.n_featuremaps, max_featuremaps),
        )

        self.model = nn.Sequential()

        for i in range(self.n_up):
            scale_factor = 2 ** (self.n_up - 1 - i)
            for j in range(n_layers):
                self.model.add_module(
                    "Decoder_{0}x{0}_{1}".format(4 * 2**i, j),
                    BottleneckResidualBlock(
                        min(scale_factor * self.n_featuremaps, max_featuremaps),
                        min(scale_factor * self.n_featuremaps, max_featuremaps),
                        norm=norm,
                        act=self.act,
                    ),
                )
            self.model.add_module(
                "Decoder_Up{}".format(i + 1),
                BottleneckResidualBlock(
                    min(scale_factor * self.n_featuremaps, max_featuremaps),
                    min(scale_factor * self.n_featuremaps // 2, max_featuremaps),
                    resample="up",
                    norm=norm,
                    act=self.act,
                ),
            )

        for j in range(n_layers - 1):
            self.model.add_module(
                "Decoder_{0}x{0}_{1}".format(4 * 2**self.n_up, j),
                BottleneckResidualBlock(
                    self.n_featuremaps // 2,
                    self.n_featuremaps // 2,
                    norm=norm,
                    act=self.act,
                ),
            )
        self.model.add_module(
            "Decoder_Out",
            nn.Conv2d(
                self.n_featuremaps // 2,
                image_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        self.apply(init_weights)

    def forward(self, encoding, cl):
        out = self.fc(torch.cat((encoding, cl), 1))
        out = self.model(
            out.reshape(
                cl.shape[0],
                min(2 ** (self.n_up - 1) * self.n_featuremaps, self.max_featuremaps),
                4,
                4,
            )
        )

        return out


class DeepEncoder(nn.Module):
    def __init__(
        self,
        cl_dim,
        n_featuremaps=N_FEATUREMAPS,
        norm="layernorm",
        latent_dim=LATENT_DIM,
        image_channels=3,
        image_size=64,
        max_featuremaps=512,
        n_layers=6,
        **kwargs,
    ):
        super().__init__()

        if log2(image_size) % 1 != 0:
            raise ValueError("'img_size' has to be a power of 2.")

        if log2(n_featuremaps) % 1 != 0:
            raise ValueError("'n_featuremaps' has to be a power of 2.")

        if not n_layers > 1:
            raise ValueError("'n_layers' should be greater than 1.")

        self.n_featuremaps = n_featuremaps
        self.img_size = image_size
        self.n_down = floor(log2(self.img_size // 4))
        self.act = nn.LeakyReLU()

        self.model = nn.Sequential()
        self.model.add_module(
            "Encoder_In",
            nn.Conv2d(
                image_channels + cl_dim,
                self.n_featuremaps // 2,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        for j in range(n_layers - 1):
            self.model.add_module(
                "Encoder_{0}x{0}_{1}".format(self.img_size, j),
                BottleneckResidualBlock(
                    self.n_featuremaps // 2,
                    self.n_featuremaps // 2,
                    norm=norm,
                    act=self.act,
                    input_spatial_size=[self.img_size] * 2,
                ),
            )

        for i in range(self.n_down):
            scale_factor = 2**i
            self.model.add_module(
                "Encoder_Down{}".format(i + 1),
                BottleneckResidualBlock(
                    min(scale_factor * self.n_featuremaps // 2, max_featuremaps),
                    min(scale_factor * self.n_featuremaps, max_featuremaps),
                    resample="down",
                    norm=norm,
                    act=self.act,
                    input_spatial_size=[self.img_size // 2**i] * 2,
                ),
            )

            for j in range(n_layers):
                self.model.add_module(
                    "Encoder_{0}x{0}_{1}".format(self.img_size // 2 ** (i + 1), j),
                    BottleneckResidualBlock(
                        min(scale_factor * self.n_featuremaps, max_featuremaps),
                        min(scale_factor * self.n_featuremaps, max_featuremaps),
                        norm=norm,
                        act=self.act,
                        input_spatial_size=[self.img_size // 2 ** (i + 1)] * 2,
                    ),
                )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=4
                * 4
                * min(2 ** (self.n_down - 1) * self.n_featuremaps, max_featuremaps),
                out_features=latent_dim,
            ),
        )

        self.apply(init_weights)

    def forward(self, x, cl):
        cl = cl.unsqueeze(-1).unsqueeze(-1)
        cl = cl.expand(list(cl.shape)[:2] + [x.shape[2], x.shape[3]])
        x = torch.cat((x, cl), 1)

        x = self.model(x)
        encoding = self.fc(x)

        return encoding


class GoodResNetCAE(BaseCAE):
    def __init__(
        self,
        cl_dim: int,
        image_channels: int = 3,
        image_size: int = 32,
        n_featuremaps: int = N_FEATUREMAPS,
        latent_dim: int = LATENT_DIM,
        *args: Tuple[Any],
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = GoodEncoder(
            cl_dim=cl_dim,
            n_featuremaps=n_featuremaps,
            latent_dim=latent_dim,
            image_channels=image_channels,
            image_size=image_size,
            **kwargs,
        )

        self.decoder = GoodDecoder(
            cl_dim=cl_dim,
            n_featuremaps=n_featuremaps,
            latent_dim=latent_dim,
            image_channels=image_channels,
            image_size=image_size,
            **kwargs,
        )

    def _encode(self, x: Tensor, cl: Tensor) -> Tensor:
        return self.encoder(x, cl)

    def _decode(self, encoding: Tensor, cl: Tensor) -> Tensor:
        return self.decoder(encoding, cl)


class GoodDecoder(nn.Module):
    def __init__(
        self,
        cl_dim,
        n_featuremaps=N_FEATUREMAPS,
        latent_dim=LATENT_DIM,
        norm="batchnorm",
        image_channels=3,
        act=nn.ReLU(),
        **kwargs,
    ):
        super().__init__()

        self.log = logging.getLogger("GoodDecoder")

        if "image_size" in kwargs and kwargs["image_size"] != 64:
            self.log.warn(
                "{} only works with an image size of 64x64".format(
                    self.__class__.__name__
                )
            )

        self.latent_dim = latent_dim
        self.image_channels = image_channels
        self.n_featuremaps = n_featuremaps

        self.fc = nn.Linear(cl_dim + self.latent_dim, 4 * 4 * 8 * self.n_featuremaps)

        self.model = nn.Sequential(
            ResidualBlock(
                8 * self.n_featuremaps,
                8 * self.n_featuremaps,
                resample="up",
                act=act,
                norm=norm,
            ),
            ResidualBlock(
                8 * self.n_featuremaps,
                4 * self.n_featuremaps,
                resample="up",
                act=act,
                norm=norm,
            ),
            ResidualBlock(
                4 * self.n_featuremaps,
                2 * self.n_featuremaps,
                resample="up",
                act=act,
                norm=norm,
            ),
            ResidualBlock(
                2 * self.n_featuremaps,
                1 * self.n_featuremaps,
                resample="up",
                act=act,
                norm=norm,
            ),
            nn.BatchNorm2d(self.n_featuremaps),
            act,
            nn.Conv2d(
                self.n_featuremaps, image_channels, kernel_size=3, stride=1, padding=1
            ),
        )

        self.apply(init_weights)

    def forward(self, encoding, cl):
        out = self.fc(torch.cat((encoding, cl), 1))
        out = self.model(out.reshape(cl.shape[0], 8 * self.n_featuremaps, 4, 4))
        return out


class GoodEncoder(nn.Module):
    def __init__(
        self,
        cl_dim,
        n_featuremaps=N_FEATUREMAPS,
        norm="layernorm",
        latent_dim=LATENT_DIM,
        image_channels=3,
        act=nn.LeakyReLU(),
        **kwargs,
    ):
        super().__init__()

        if "image_size" in kwargs and kwargs["image_size"] != 64:
            raise ValueError(
                "{} only works with an image size of 64x64".format(
                    self.__class__.__name__
                )
            )

        self.model = nn.Sequential(
            nn.Conv2d(
                cl_dim + image_channels,
                n_featuremaps,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            ResidualBlock(
                n_featuremaps,
                2 * n_featuremaps,
                resample="down",
                norm=norm,
                act=act,
                input_spatial_size=[64, 64],
            ),
            ResidualBlock(
                2 * n_featuremaps,
                4 * n_featuremaps,
                resample="down",
                norm=norm,
                act=act,
                input_spatial_size=[32, 32],
            ),
            ResidualBlock(
                4 * n_featuremaps,
                8 * n_featuremaps,
                resample="down",
                norm=norm,
                act=act,
                input_spatial_size=[16, 16],
            ),
            ResidualBlock(
                8 * n_featuremaps,
                8 * n_featuremaps,
                resample="down",
                norm=norm,
                act=act,
                input_spatial_size=[8, 8],
            ),
        )

        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(4 * 4 * 8 * n_featuremaps, latent_dim)
        )

        self.apply(init_weights)

    def forward(self, x, cl):
        cl = cl.unsqueeze(-1).unsqueeze(-1)
        cl = cl.expand(list(cl.shape)[:2] + [x.shape[2], x.shape[3]])
        x = torch.cat((x, cl), 1)

        x = self.model(x)
        encoding = self.fc(x)

        return encoding
