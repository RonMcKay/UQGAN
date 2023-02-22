import torch.nn as nn

from bnn_models import BNN_MODELS

from .base import BaseClassifier


class SmallClassifier(BaseClassifier):
    def __init__(self, cl_dim, image_channels=1, image_size=28, **kwargs):
        super().__init__(cl_dim=cl_dim, **kwargs)

        if self.method == "bayes":
            raise ValueError(
                "Method 'bayes' needs to be used together with "
                "one of the bayesian models. "
                f"Valid options are: [{', '.join(BNN_MODELS)}]"
            )

        self.mc_dropout: float = kwargs.get("mc_dropout", 0.0)

        if (image_size / 4.0) % 1 != 0:
            raise ValueError("'image_size' should be divisible by 4.")

        self.cl_dim = cl_dim
        self.image_channels = image_channels
        self.image_size = image_size

        self.save_hyperparameters()

        if self.mc_dropout > 0:
            self.model = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.image_channels,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.Dropout(p=self.mc_dropout),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
                ),
                nn.Dropout(p=self.mc_dropout),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
                ),
                nn.Dropout(p=self.mc_dropout),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(self.image_size // 4),
                nn.Flatten(),
            )

            self.fc = nn.Sequential(
                nn.Linear(
                    in_features=64 * (self.image_size // 4) ** 2, out_features=512
                ),
                nn.Dropout(p=self.mc_dropout),
                nn.ReLU(),
                nn.Linear(in_features=512, out_features=self.cl_dim),
            )
        else:
            self.model = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.image_channels,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
                ),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(self.image_size // 4),
                nn.Flatten(),
            )

            self.fc = nn.Sequential(
                nn.Linear(
                    in_features=64 * (self.image_size // 4) ** 2, out_features=512
                ),
                nn.ReLU(),
                nn.Linear(in_features=512, out_features=self.cl_dim),
            )

    def _forward(self, x):
        out = self.model(x)
        return self.fc(out)
