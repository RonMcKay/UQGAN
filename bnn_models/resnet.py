from math import ceil, floor, log2

import bnn
import torch
import torch.nn as nn

from cls_models.base import BaseClassifier
from utils import init_weights

N_FEATUREMAPS = 32


class BSubpixelConv2D(bnn.BayesianLayer):
    def __init__(
        self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.model = bnn.Sequential(
            bnn.BConv2d(
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


class BConvMeanPool(bnn.BayesianLayer):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, bias=True
    ):
        super().__init__()

        self.model = bnn.Sequential(
            bnn.BConv2d(
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


class BMeanPoolConv(bnn.BayesianLayer):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, bias=True
    ):
        super().__init__()

        self.model = bnn.Sequential(
            nn.AvgPool2d(2),
            bnn.BConv2d(
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


class BUpsampleConv(bnn.BayesianLayer):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, bias=True
    ):
        super().__init__()

        self.model = bnn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            bnn.BConv2d(
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


class BBottleneckResidualBlock(bnn.BayesianLayer):
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
            self.shortcut = bnn.BConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2,
                padding=0,
            )
            self.conv1 = bnn.BConv2d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
            # compute padding for given kernel size so that spatial size halves
            # with stride of 2 this assumes that image size is a power of 2
            padding = floor(kernel_size / 2.0)
            self.conv1b = bnn.BConv2d(
                in_channels=in_channels // 2,
                out_channels=out_channels // 2,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                bias=True,
            )
            self.conv2 = bnn.BConv2d(
                in_channels=out_channels // 2,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
        elif resample == "up":
            self.shortcut = BSubpixelConv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
            self.conv1 = bnn.BConv2d(
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
            self.conv1b = bnn.BConvTranspose2d(
                in_channels=in_channels // 2,
                out_channels=out_channels // 2,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=True,
            )
            self.conv2 = bnn.BConv2d(
                in_channels=out_channels // 2,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
        elif resample is None:
            self.shortcut = bnn.BConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
            self.conv1 = bnn.BConv2d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
            self.conv1b = bnn.BConv2d(
                in_channels=in_channels // 2,
                out_channels=out_channels // 2,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
            )
            self.conv2 = bnn.BConv2d(
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
        kl = torch.tensor(0.0, device=x.device, requires_grad=True)

        shortcut, _kl = self.shortcut(x)
        kl = kl + _kl

        x = self.act(x)
        x, _kl = self.conv1(x)
        kl = kl + _kl

        x = self.act(x)
        x, _kl = self.conv1b(x)
        kl = kl + _kl

        x = self.act(x)
        x, _kl = self.conv2(x)
        kl = kl + _kl

        x = self.norm(x)
        return shortcut + 0.3 * x, kl


class BResidualBlock(nn.Module):
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
            self.shortcut = BMeanPoolConv(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )
            self.conv1 = bnn.BConv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            self.conv2 = BConvMeanPool(
                in_channels, out_channels, kernel_size, stride, padding
            )
        elif resample == "up":
            self.shortcut = BUpsampleConv(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )
            self.conv1 = BUpsampleConv(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            self.conv2 = bnn.BConv2d(
                out_channels, out_channels, kernel_size, stride=stride, padding=padding
            )
        elif resample is None:
            self.shortcut = bnn.BConv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )
            self.conv1 = bnn.BConv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            self.conv2 = bnn.BConv2d(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding
            )
        else:
            raise ValueError(
                (
                    "Unknown resample type '{resample}'. "
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
        kl = torch.tensor(0.0, device=x.device, requires_grad=True)

        shortcut, _kl = self.shortcut(x)
        kl = kl + _kl

        x = self.norm1(x)
        x = self.act(x)
        x, _kl = self.conv1(x)
        kl = kl + _kl

        x = self.norm2(x)
        x = self.act(x)
        x, _kl = self.conv2(x)
        kl = kl + _kl

        return shortcut + x


class BClassifier(BaseClassifier):
    def __init__(
        self,
        cl_dim,
        n_featuremaps=N_FEATUREMAPS,
        norm="layernorm",
        image_channels=3,
        image_size=64,
        max_featuremaps=512,
        **kwargs,
    ):
        super().__init__(
            method="bayes", cl_dim=cl_dim, mc_samples=kwargs.get("mc_samples", 1)
        )

        if log2(image_size) % 1 != 0:
            raise ValueError("'img_size' has to be a power of 2.")

        if log2(n_featuremaps) % 1 != 0:
            raise ValueError("'n_featuremaps' has to be a power of 2.")

        self.cl_dim = cl_dim
        self.n_featuremaps = n_featuremaps
        self.img_size = image_size
        self.act = nn.LeakyReLU()
        self.n_down = floor(log2(self.img_size // 4))

        self.model = bnn.Sequential()
        self.model.add_module(
            "Classifier_In",
            bnn.BConv2d(
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
                "Classifier_{0}x{0}".format(self.img_size // scale_factor),
                BResidualBlock(
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
            "Classifier_{0}x{0}".format(self.img_size // scale_factor),
            BResidualBlock(
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
            bnn.BLinear(
                in_features=4**2
                * min(2 ** (self.n_down - 1) * self.n_featuremaps, max_featuremaps),
                out_features=self.cl_dim,
            ),
        )

        self.apply(init_weights)

        self.save_hyperparameters()

    def _forward(self, x):

        x, kl = self.model(x)
        out, _kl = self.fc(x)
        kl = kl + _kl

        return out, kl
