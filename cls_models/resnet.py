from math import floor, log2

import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd

from bnn_models import BNN_MODELS
from gan_models.resnet import ResidualBlock

from .base import BaseClassifier

N_FEATUREMAPS = 32


def init_weights(m):
    if isinstance(m, _ConvNd):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, _BatchNorm) or isinstance(m, nn.LayerNorm):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class Classifier(BaseClassifier):
    def __init__(
        self,
        cl_dim,
        n_featuremaps=N_FEATUREMAPS,
        blocks=1,
        norm="layernorm",
        image_channels=3,
        image_size=64,
        max_featuremaps=512,
        mc_dropout=0.0,
        **kwargs,
    ):
        super().__init__(cl_dim=cl_dim, **kwargs)

        if self.method == "bayes":
            raise ValueError(
                "Method 'bayes' needs to be used together with "
                "one of the bayesian models. "
                f"Valid options are: [{', '.join(BNN_MODELS)}]"
            )

        if log2(image_size) % 1 != 0:
            raise ValueError("'img_size' has to be a power of 2.")

        if log2(n_featuremaps) % 1 != 0:
            raise ValueError("'n_featuremaps' has to be a power of 2.")

        self.cl_dim = cl_dim
        self.n_featuremaps = n_featuremaps
        self.img_size = image_size
        self.act = nn.LeakyReLU()
        self.mc_dropout = mc_dropout
        self.n_down = floor(log2(self.img_size // 4))

        self.save_hyperparameters()

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
        if mc_dropout > 0:
            self.model.add_module(
                "Classifier_In",
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=image_channels,
                        out_channels=self.n_featuremaps,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.Dropout(p=mc_dropout),
                ),
            )
        else:
            self.model.add_module(
                "Classifier_In",
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
                "Classifier_{0}x{0}".format(self.img_size // scale_factor)
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
                    mc_dropout=mc_dropout,
                ),
            )
            for j in range(self.blocks[i] - 1):
                self.model.add_module(
                    "Classifier_{0}x{0}_{1}".format(
                        self.img_size // scale_factor, j + 1
                    ),
                    ResidualBlock(
                        in_channels=min(
                            2 * scale_factor * self.n_featuremaps, max_featuremaps
                        ),
                        out_channels=min(
                            2 * scale_factor * self.n_featuremaps, max_featuremaps
                        ),
                        resample=None,
                        norm=norm,
                        act=self.act,
                        input_spatial_size=[self.img_size // scale_factor // 2] * 2,
                        mc_dropout=mc_dropout,
                    ),
                )

        scale_factor = 2 ** (self.n_down - 1)
        self.model.add_module(
            "Classifier_{0}x{0}".format(self.img_size // scale_factor)
            + ("_0" if self.blocks[-1] > 1 else ""),
            ResidualBlock(
                in_channels=min(scale_factor * self.n_featuremaps, max_featuremaps),
                out_channels=min(scale_factor * self.n_featuremaps, max_featuremaps),
                resample="down",
                norm=norm,
                act=self.act,
                input_spatial_size=[self.img_size // scale_factor] * 2,
                mc_dropout=mc_dropout,
            ),
        )
        for j in range(self.blocks[-1] - 1):
            self.model.add_module(
                "Classifier_{0}x{0}_{1}".format(self.img_size // scale_factor, j + 1),
                ResidualBlock(
                    in_channels=min(scale_factor * self.n_featuremaps, max_featuremaps),
                    out_channels=min(
                        scale_factor * self.n_featuremaps, max_featuremaps
                    ),
                    resample=None,
                    norm=norm,
                    act=self.act,
                    input_spatial_size=[self.img_size // scale_factor // 2] * 2,
                    mc_dropout=mc_dropout,
                ),
            )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=4**2
                * min(2 ** (self.n_down - 1) * self.n_featuremaps, max_featuremaps),
                out_features=self.cl_dim,
            ),
        )

        self.apply(init_weights)

    def _forward(self, x):

        x = self.model(x)
        out = self.fc(x)

        return out
