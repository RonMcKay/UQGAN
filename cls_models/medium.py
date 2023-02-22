import torch.nn as nn

from bnn_models import BNN_MODELS

from .base import BaseClassifier


class MediumClassifier(BaseClassifier):
    def __init__(
        self, cl_dim, image_channels=1, image_size=32, num_channels=128, **kwargs
    ):
        super().__init__(cl_dim=cl_dim, **kwargs)

        if self.method == "bayes":
            raise ValueError(
                "Method 'bayes' needs to be used together with "
                "one of the bayesian models. "
                f"Valid options are: [{', '.join(BNN_MODELS)}]"
            )

        self.mc_dropout: float = kwargs.get("mc_dropout", 0.0)

        if (image_size / 8.0) % 1 != 0:
            raise ValueError("'image_size' should be divisible by 8.")

        self.cl_dim = cl_dim
        self.image_channels = image_channels
        self.image_size = image_size
        self.num_channels = num_channels

        self.save_hyperparameters()

        self.model = nn.Sequential(
            # Downsample 1
            self.conv_block_down(
                in_channels=self.image_channels,
                out_channels=self.num_channels // 4,
                dropout=self.mc_dropout,
            ),
            # Downsample 2
            self.conv_block_down(
                in_channels=self.num_channels // 4,
                out_channels=self.num_channels // 2,
                dropout=self.mc_dropout,
            ),
            # Downsample 3
            self.conv_block_down(
                in_channels=self.num_channels // 2,
                out_channels=self.num_channels,
                dropout=self.mc_dropout,
            ),
            nn.AdaptiveAvgPool2d(self.image_size // 8),
            # Embedding output
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(
                in_features=self.num_channels * (self.image_size // 8) ** 2,
                out_features=self.cl_dim,
            ),
        )

    def _forward(self, x):
        out = self.model(x)
        return self.fc(out)

    def conv_block_down(self, in_channels, out_channels, dropout: float = 0.0):
        block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        if dropout > 0:
            block.append(nn.Dropout(p=dropout))

        block.append(nn.BatchNorm2d(out_channels))
        block.append(
            nn.LeakyReLU(0.2),
        )
        return block
