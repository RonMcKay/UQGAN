# This is a PyTorch reimplementation of the code to the paper
# 'Improved Training of Wasserstein GANs'
# https://github.com/igul222/improved_wgan_training

import logging
from math import ceil, floor, log2

from pytorch_lightning import LightningModule
import torch
from torch.distributions.uniform import Uniform
import torch.nn as nn

from utils import init_weights

LATENT_DIM = 128
N_FEATUREMAPS = 32


class SubpixelConv2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=True,
        mc_dropout=0.0,
    ):
        super().__init__()

        if mc_dropout > 0:
            self.model = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels * 4,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                nn.Dropout(p=mc_dropout),
                nn.PixelShuffle(2),
            )
        else:
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
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias=True,
        mc_dropout=0.0,
    ):
        super().__init__()

        if mc_dropout > 0:
            self.model = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                nn.Dropout(p=mc_dropout),
                nn.AvgPool2d(2),
            )
        else:
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
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias=True,
        mc_dropout=0.0,
    ):
        super().__init__()

        if mc_dropout > 0:
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
                nn.Dropout(p=mc_dropout),
            )
        else:
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
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias=True,
        mc_dropout=0.0,
    ):
        super().__init__()

        if mc_dropout > 0:
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
                nn.Dropout(p=mc_dropout),
            )
        else:
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
        mc_dropout=0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        self.mc_dropout = mc_dropout

        if mc_dropout > 0:
            self.dropout = nn.Dropout(p=mc_dropout)

        if resample == "down":
            if mc_dropout > 0:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=2,
                        padding=0,
                    ),
                    nn.Dropout(p=mc_dropout),
                )
            else:
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
            # compute padding for given kernel size so that
            # spatial size halves with stride of 2
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
                mc_dropout=mc_dropout,
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
        if self.mc_dropout > 0:
            x = self.dropout(x)
        x = self.act(x)
        x = self.conv1b(x)
        if self.mc_dropout > 0:
            x = self.dropout(x)
        x = self.act(x)
        x = self.conv2(x)
        if self.mc_dropout > 0:
            x = self.dropout(x)
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
        mc_dropout=0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act

        if mc_dropout > 0:
            self.dropout = nn.Dropout(p=mc_dropout)

        if resample == "down":
            self.shortcut = MeanPoolConv(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                mc_dropout=mc_dropout,
            )
            if mc_dropout > 0:
                self.conv1 = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=False,
                    ),
                    self.dropout,
                )
            else:
                self.conv1 = nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                )
            self.conv2 = ConvMeanPool(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                mc_dropout=mc_dropout,
            )
        elif resample == "up":
            self.shortcut = UpsampleConv(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                mc_dropout=mc_dropout,
            )
            self.conv1 = UpsampleConv(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
                mc_dropout=mc_dropout,
            )
            if mc_dropout > 0:
                self.conv2 = nn.Sequential(
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    self.dropout,
                )
            else:
                self.conv2 = nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                )
        elif resample is None:
            if mc_dropout > 0:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_channels, out_channels, kernel_size=1, stride=1, padding=0
                    ),
                    self.dropout,
                )
                self.conv1 = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=False,
                    ),
                    self.dropout,
                )
                self.conv2 = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    self.dropout,
                )
            else:
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
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
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


class Generator(LightningModule):
    def __init__(
        self,
        cl_dim=None,
        n_featuremaps=N_FEATUREMAPS,
        latent_dim=LATENT_DIM,
        norm="batchnorm",
        image_channels=3,
        image_size=64,
        max_featuremaps=512,
        conditional=True,
        **kwargs,
    ):
        super().__init__()

        if log2(image_size) % 1 != 0:
            raise ValueError("'img_size' has to be a power of 2.")

        if log2(n_featuremaps) % 1 != 0:
            raise ValueError("'n_featuremaps' has to be a power of 2.")

        if conditional and cl_dim is None:
            raise ValueError(
                "When conditional is set to 'true', 'cl_dim' must be specified."
            )

        self.n_featuremaps = n_featuremaps
        self.latent_dim = latent_dim
        self.latent_distribution = Uniform(0, 1)
        self.img_size = image_size
        self.max_featuremaps = max_featuremaps
        self.act = nn.ReLU()
        self.conditional = conditional
        self.n_up = floor(log2(self.img_size // 4))

        if self.conditional:
            self.fc = nn.Linear(
                cl_dim + self.latent_dim,
                4**2
                * min(2 ** (self.n_up - 1) * self.n_featuremaps, max_featuremaps),
            )
        else:
            self.fc = nn.Linear(
                self.latent_dim,
                4**2
                * min(2 ** (self.n_up - 1) * self.n_featuremaps, max_featuremaps),
            )

        self.model = nn.Sequential()
        self.model.add_module(
            "Generator_4x4",
            ResidualBlock(
                min(2 ** (self.n_up - 1) * self.n_featuremaps, max_featuremaps),
                min(2 ** (self.n_up - 1) * self.n_featuremaps, max_featuremaps),
                resample="up",
                norm=norm,
                act=self.act,
            ),
        )
        for i in range(1, self.n_up):
            scale_factor = 2 ** (self.n_up - i)
            self.model.add_module(
                "Generator_{0}x{0}".format(self.img_size // scale_factor),
                ResidualBlock(
                    min(scale_factor * self.n_featuremaps, max_featuremaps),
                    min(scale_factor // 2 * self.n_featuremaps, max_featuremaps),
                    resample="up",
                    norm=norm,
                    act=self.act,
                ),
            )

        self.model.add_module(
            "Generator_Norm_Out",
            nn.BatchNorm2d(num_features=self.n_featuremaps),
        )
        self.model.add_module(
            "Generator_Act_Out",
            self.act,
        )
        self.model.add_module(
            "Generator_Conv_Out",
            nn.Conv2d(
                in_channels=self.n_featuremaps,
                out_channels=image_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.model.add_module(
            "Generator_Tanh_Out",
            nn.Sigmoid(),
        )

        self.apply(init_weights)

        self.save_hyperparameters()

    def forward(self, *args, **kwargs):
        if self.conditional:
            return self._forward_conditional(*args, **kwargs)

    def _forward_conditional(self, cl, z=None):
        if z is None:
            latent_code = self.latent_distribution.sample(
                (cl.shape[0], self.latent_dim)
            ).to(cl.device)
        else:
            latent_code = z

        out = self.fc(torch.cat((cl, latent_code), 1))
        out = self.model(
            out.reshape(
                cl.shape[0],
                min(2 ** (self.n_up - 1) * self.n_featuremaps, self.max_featuremaps),
                4,
                4,
            )
        )
        return out

    def _forward(self, num_samples=None, z=None):
        if z is None:
            if num_samples is None:
                raise ValueError("Either 'num_samples' or 'z' should be specified.")
            latent_code = self.latent_distribution.sample(
                (num_samples, self.latent_dim)
            ).to(next(self.parameters()).device)
        else:
            latent_code = z

        out = self.fc(latent_code)
        out = self.model(
            out.reshape(
                z.shape[0] if num_samples is None else num_samples,
                min(2 ** (self.n_up - 1) * self.n_featuremaps, self.max_featuremaps),
                4,
                4,
            )
        )
        return out


class Discriminator(LightningModule):
    def __init__(
        self,
        cl_dim=None,
        n_featuremaps=N_FEATUREMAPS,
        norm="layernorm",
        image_channels=3,
        image_size=64,
        max_featuremaps=512,
        conditional=True,
        **kwargs,
    ):
        super().__init__()

        if log2(image_size) % 1 != 0:
            raise ValueError("'img_size' has to be a power of 2.")

        if log2(n_featuremaps) % 1 != 0:
            raise ValueError("'n_featuremaps' has to be a power of 2.")

        if conditional and cl_dim is None:
            raise ValueError(
                "When conditional is set to 'true', 'cl_dim' must be specified."
            )

        self.n_featuremaps = n_featuremaps
        self.img_size = image_size
        self.act = nn.LeakyReLU()
        self.conditional = conditional
        self.n_down = floor(log2(self.img_size // 4))

        self.model = nn.Sequential()
        if self.conditional:
            self.model.add_module(
                "Discriminator_In",
                nn.Conv2d(
                    in_channels=image_channels + cl_dim,
                    out_channels=self.n_featuremaps,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )
        else:
            self.model.add_module(
                "Discriminator_In",
                nn.Conv2d(
                    in_channels=image_channels,
                    out_channels=self.n_featuremaps,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )
        for i in range(self.n_down - 1):
            scale_factor = 2**i
            self.model.add_module(
                "Discriminator_{0}x{0}".format(self.img_size // scale_factor),
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

        scale_factor = 2 ** (self.n_down - 1)
        self.model.add_module(
            "Discriminator_{0}x{0}".format(self.img_size // scale_factor),
            ResidualBlock(
                in_channels=min(scale_factor * self.n_featuremaps, max_featuremaps),
                out_channels=min(scale_factor * self.n_featuremaps, max_featuremaps),
                resample="down",
                norm=norm,
                act=self.act,
                input_spatial_size=[self.img_size // scale_factor] * 2,
            ),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=4**2
                * min(2 ** (self.n_down - 1) * self.n_featuremaps, max_featuremaps),
                out_features=1,
            ),
        )

        self.apply(init_weights)

        self.save_hyperparameters()

    def forward(self, *args, **kwargs):
        if self.conditional:
            return self._forward_conditional(*args, **kwargs)
        else:
            return self._forward(*args, **kwargs)

    def _forward_conditional(self, x, cl):
        cl = cl.unsqueeze(-1).unsqueeze(-1)
        cl = cl.expand(list(cl.shape)[:2] + [x.shape[2], x.shape[3]])
        x = torch.cat((x, cl), 1)

        x = self.model(x)
        out_dc = self.fc(x)

        return out_dc

    def _forward(self, x):
        x = self.model(x)
        out_dc = self.fc(x)

        return out_dc


class DeepGenerator(LightningModule):
    def __init__(
        self,
        cl_dim=None,
        n_featuremaps=N_FEATUREMAPS,
        latent_dim=LATENT_DIM,
        norm="batchnorm",
        image_channels=3,
        image_size=64,
        max_featuremaps=512,
        n_layers=6,
        conditional=True,
        **kwargs,
    ):
        super().__init__()

        if log2(image_size) % 1 != 0:
            raise ValueError("'img_size' has to be a power of 2.")

        if log2(n_featuremaps) % 1 != 0:
            raise ValueError("'n_featuremaps' has to be a power of 2.")

        if not n_layers > 1:
            raise ValueError("'n_layers' should be greater than 1.")

        if conditional and cl_dim is None:
            raise ValueError(
                "When conditional is set to 'true', 'cl_dim' must be specified."
            )

        self.n_featuremaps = n_featuremaps
        self.latent_dim = latent_dim
        self.latent_distribution = Uniform(0, 1)
        self.img_size = image_size
        self.max_featuremaps = max_featuremaps
        self.n_up = floor(log2(self.img_size // 4))
        self.act = nn.ReLU()
        self.conditional = conditional

        if self.conditional:
            self.fc = nn.Linear(
                cl_dim + self.latent_dim,
                4**2
                * min(2 ** (self.n_up - 1) * self.n_featuremaps, max_featuremaps),
            )
        else:
            self.fc = nn.Linear(
                self.latent_dim,
                4**2
                * min(2 ** (self.n_up - 1) * self.n_featuremaps, max_featuremaps),
            )

        self.model = nn.Sequential()

        for i in range(self.n_up):
            scale_factor = 2 ** (self.n_up - 1 - i)
            for j in range(n_layers):
                self.model.add_module(
                    "Generator_{0}x{0}_{1}".format(4 * 2**i, j),
                    BottleneckResidualBlock(
                        min(scale_factor * self.n_featuremaps, max_featuremaps),
                        min(scale_factor * self.n_featuremaps, max_featuremaps),
                        norm=norm,
                        act=self.act,
                    ),
                )
            self.model.add_module(
                "Generator_Up{}".format(i + 1),
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
                "Generator_{0}x{0}_{1}".format(4 * 2**self.n_up, j),
                BottleneckResidualBlock(
                    self.n_featuremaps // 2,
                    self.n_featuremaps // 2,
                    norm=norm,
                    act=self.act,
                ),
            )
        self.model.add_module(
            "Generator_Out",
            nn.Conv2d(
                self.n_featuremaps // 2,
                image_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        self.model.add_module(
            "Generator_Tanh_Out",
            nn.Sigmoid(),
        )

        self.apply(init_weights)

        self.save_hyperparameters()

    def forward(self, *args, **kwargs):
        if self.conditional:
            return self._forward_conditional(*args, **kwargs)
        else:
            return self._forward(*args, **kwargs)

    def _forward_conditional(self, cl, z=None):
        if z is None:
            latent_code = self.latent_distribution.sample(
                (cl.shape[0], self.latent_dim)
            ).to(cl.device)
        else:
            latent_code = z

        out = self.fc(torch.cat((cl, latent_code), 1))
        out = self.model(
            out.reshape(
                cl.shape[0],
                min(2 ** (self.n_up - 1) * self.n_featuremaps, self.max_featuremaps),
                4,
                4,
            )
        )

        return out

    def _forward(self, num_samples=None, z=None):
        if z is None:
            if num_samples is None:
                raise ValueError("Either 'num_samples' or 'z' should be specified.")
            latent_code = self.latent_distribution.sample(
                (num_samples, self.latent_dim)
            ).to(next(self.parameters()).device)
        else:
            latent_code = z

        out = self.fc(latent_code)
        out = self.model(
            out.reshape(
                z.shape[0] if num_samples is None else num_samples,
                min(2 ** (self.n_up - 1) * self.n_featuremaps, self.max_featuremaps),
                4,
                4,
            )
        )

        return out


class DeepDiscriminator(LightningModule):
    def __init__(
        self,
        cl_dim=None,
        n_featuremaps=N_FEATUREMAPS,
        norm="layernorm",
        image_channels=3,
        image_size=64,
        max_featuremaps=512,
        n_layers=6,
        conditional=True,
        **kwargs,
    ):
        super().__init__()

        if log2(image_size) % 1 != 0:
            raise ValueError("'img_size' has to be a power of 2.")

        if log2(n_featuremaps) % 1 != 0:
            raise ValueError("'n_featuremaps' has to be a power of 2.")

        if not n_layers > 1:
            raise ValueError("'n_layers' should be greater than 1.")

        if conditional and cl_dim is None:
            raise ValueError(
                "When conditional is set to 'true', 'cl_dim' must be specified."
            )

        self.n_featuremaps = n_featuremaps
        self.img_size = image_size
        self.n_down = floor(log2(self.img_size // 4))
        self.act = nn.LeakyReLU()
        self.conditional = conditional

        self.model = nn.Sequential()
        if self.conditional:
            self.model.add_module(
                "Discriminator_In",
                nn.Conv2d(
                    image_channels + cl_dim,
                    self.n_featuremaps // 2,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
            )
        else:
            self.model.add_module(
                "Discriminator_In",
                nn.Conv2d(
                    image_channels,
                    self.n_featuremaps // 2,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
            )

        for j in range(n_layers - 1):
            self.model.add_module(
                "Discriminator_{0}x{0}_{1}".format(self.img_size, j),
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
                "Discriminator_Down{}".format(i + 1),
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
                    "Discriminator_{0}x{0}_{1}".format(
                        self.img_size // 2 ** (i + 1), j
                    ),
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
                out_features=1,
            ),
        )

        self.apply(init_weights)

        self.save_hyperparameters()

    def forward(self, *args, **kwargs):
        if self.conditional:
            return self._forward_conditional(*args, **kwargs)
        else:
            return self._forward(*args, **kwargs)

    def _forward_conditional(self, x, cl):
        cl = cl.unsqueeze(-1).unsqueeze(-1)
        cl = cl.expand(list(cl.shape)[:2] + [x.shape[2], x.shape[3]])
        x = torch.cat((x, cl), 1)

        x = self.model(x)
        out_dc = self.fc(x)

        return out_dc

    def _forward(self, x):
        x = self.model(x)
        out_dc = self.fc(x)

        return out_dc


class GoodGenerator(LightningModule):
    def __init__(
        self,
        cl_dim=None,
        n_featuremaps=N_FEATUREMAPS,
        latent_dim=LATENT_DIM,
        norm="batchnorm",
        image_channels=3,
        conditional=True,
        **kwargs,
    ):
        super().__init__()

        self.log = logging.getLogger("GoodGenerator")

        if "image_size" in kwargs and kwargs["image_size"] != 64:
            self.log.warn(
                "{} only works with an image size of 64x64".format(
                    self.__class__.__name__
                )
            )

        if conditional and cl_dim is None:
            raise ValueError(
                "When conditional is set to 'true', 'cl_dim' must be specified."
            )

        self.latent_dim = latent_dim
        self.latent_distribution = Uniform(0, 1)
        self.image_channels = image_channels
        self.n_featuremaps = n_featuremaps
        self.conditional = conditional
        self.act = nn.ReLU()

        if self.conditional:
            self.fc = nn.Linear(
                cl_dim + self.latent_dim, 4 * 4 * 8 * self.n_featuremaps
            )
        else:
            self.fc = nn.Linear(self.latent_dim, 4 * 4 * 8 * self.n_featuremaps)

        self.model = nn.Sequential(
            ResidualBlock(
                8 * self.n_featuremaps,
                8 * self.n_featuremaps,
                resample="up",
                act=self.act,
                norm=norm,
            ),
            ResidualBlock(
                8 * self.n_featuremaps,
                4 * self.n_featuremaps,
                resample="up",
                act=self.act,
                norm=norm,
            ),
            ResidualBlock(
                4 * self.n_featuremaps,
                2 * self.n_featuremaps,
                resample="up",
                act=self.act,
                norm=norm,
            ),
            ResidualBlock(
                2 * self.n_featuremaps,
                1 * self.n_featuremaps,
                resample="up",
                act=self.act,
                norm=norm,
            ),
            nn.BatchNorm2d(self.n_featuremaps),
            self.act,
            nn.Conv2d(
                self.n_featuremaps, image_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.Sigmoid(),
        )

        self.apply(init_weights)

        self.save_hyperparameters()

    def forward(self, *args, **kwargs):
        if self.conditional:
            return self._forward_conditional(*args, **kwargs)
        else:
            return self._forward(*args, **kwargs)

    def _forward_conditional(self, cl, z=None):
        if z is None:
            latent_code = self.latent_distribution.sample(
                (cl.shape[0], self.latent_dim)
            ).to(cl.device)
        else:
            latent_code = z

        out = self.fc(torch.cat((cl, latent_code), 1))
        out = self.model(out.reshape(cl.shape[0], 8 * self.n_featuremaps, 4, 4))
        return out

    def _forward(self, num_samples=None, z=None):
        if z is None:
            if num_samples is None:
                raise ValueError("Either 'num_samples' or 'z' should be specified.")
            latent_code = self.latent_distribution.sample(
                (num_samples, self.latent_dim)
            ).to(next(self.parameters()).device)
        else:
            latent_code = z

        out = self.fc(latent_code)
        out = self.model(
            out.reshape(
                z.shape[0] if num_samples is None else num_samples,
                8 * self.n_featuremaps,
                4,
                4,
            )
        )
        return out


class GoodDiscriminator(LightningModule):
    def __init__(
        self,
        cl_dim=None,
        n_featuremaps=N_FEATUREMAPS,
        norm="layernorm",
        image_channels=3,
        conditional=True,
        **kwargs,
    ):
        super().__init__()

        if "image_size" in kwargs and kwargs["image_size"] != 64:
            raise ValueError(
                "{} only works with an image size of 64x64".format(
                    self.__class__.__name__
                )
            )

        if conditional and cl_dim is None:
            raise ValueError(
                "When conditional is set to 'true', 'cl_dim' must be specified."
            )

        self.conditional = conditional
        self.act = nn.LeakyReLU()

        self.model = nn.Sequential()
        if self.conditional:
            self.model = nn.Sequential(
                nn.Conv2d(
                    cl_dim + image_channels if not self.acgan else image_channels,
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
                    act=self.act,
                    input_spatial_size=[64, 64],
                ),
                ResidualBlock(
                    2 * n_featuremaps,
                    4 * n_featuremaps,
                    resample="down",
                    norm=norm,
                    act=self.act,
                    input_spatial_size=[32, 32],
                ),
                ResidualBlock(
                    4 * n_featuremaps,
                    8 * n_featuremaps,
                    resample="down",
                    norm=norm,
                    act=self.act,
                    input_spatial_size=[16, 16],
                ),
                ResidualBlock(
                    8 * n_featuremaps,
                    8 * n_featuremaps,
                    resample="down",
                    norm=norm,
                    act=self.act,
                    input_spatial_size=[8, 8],
                ),
            )
        else:
            self.model = nn.Sequential(
                nn.Conv2d(
                    image_channels, n_featuremaps, kernel_size=3, stride=1, padding=1
                ),
                ResidualBlock(
                    n_featuremaps,
                    2 * n_featuremaps,
                    resample="down",
                    norm=norm,
                    act=self.act,
                    input_spatial_size=[64, 64],
                ),
                ResidualBlock(
                    2 * n_featuremaps,
                    4 * n_featuremaps,
                    resample="down",
                    norm=norm,
                    act=self.act,
                    input_spatial_size=[32, 32],
                ),
                ResidualBlock(
                    4 * n_featuremaps,
                    8 * n_featuremaps,
                    resample="down",
                    norm=norm,
                    act=self.act,
                    input_spatial_size=[16, 16],
                ),
                ResidualBlock(
                    8 * n_featuremaps,
                    8 * n_featuremaps,
                    resample="down",
                    norm=norm,
                    act=self.act,
                    input_spatial_size=[8, 8],
                ),
            )

        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(4 * 4 * 8 * n_featuremaps, 1))

        self.apply(init_weights)

        self.save_hyperparameters()

    def forward(self, *args, **kwargs):
        if self.conditional:
            return self._forward_conditional(*args, **kwargs)
        else:
            return self._forward(*args, **kwargs)

    def _forward_conditional(self, x, cl):
        cl = cl.unsqueeze(-1).unsqueeze(-1)
        cl = cl.expand(list(cl.shape)[:2] + [x.shape[2], x.shape[3]])
        x = torch.cat((x, cl), 1)

        x = self.model(x)
        out_dc = self.fc(x)

        return out_dc

    def _forward(self, x):
        x = self.model(x)
        out_dc = self.fc(x)

        return out_dc


if __name__ == "__main__":
    from nnaddons import count_parameters

    # this is for debugging purposes
    CL_DIM = 85
    N_SAMPLES = 1
    IMAGE_SIZE = 128
    N_LAYERS = 6
    print("'cl_dim': {} || 'n_samples': {}".format(CL_DIM, N_SAMPLES))

    gen = DeepGenerator(CL_DIM, image_size=IMAGE_SIZE, n_layers=N_LAYERS)
    x_tilde = gen(torch.zeros((N_SAMPLES, CL_DIM)))
    print("Generator output shape: {}".format(x_tilde.shape))
    print("Generator parameters: {}".format(count_parameters(gen)))

    disc = DeepDiscriminator(CL_DIM, image_size=IMAGE_SIZE, n_layers=N_LAYERS)
    disc_x_tilde, _ = disc(x_tilde, torch.zeros((N_SAMPLES, CL_DIM)))
    print("Discriminator output shape: {}".format(disc_x_tilde.shape))
    print("Discriminator parameters: {}".format(count_parameters(disc)))
