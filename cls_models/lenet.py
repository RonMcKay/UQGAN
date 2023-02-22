import torch
import torch.nn as nn

from bnn_models import BNN_MODELS

from .base import BaseClassifier


class LeNet(BaseClassifier):
    def __init__(
        self,
        cl_dim: int,
        image_channels: int = 1,
        image_size: int = 32,
        **kwargs,
    ) -> None:
        super().__init__(cl_dim=cl_dim, **kwargs)

        if self.method == "bayes":
            raise ValueError(
                "Method 'bayes' needs to be used together with "
                "one of the bayesian models. "
                f"Valid options are: [{', '.join(BNN_MODELS)}]"
            )

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
        self.mc_dropout: float = kwargs.get("mc_dropout", 0.0)

        self.save_hyperparameters()

        if self.mc_dropout > 0:
            self.model = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.image_channels, out_channels=6, kernel_size=5
                ),
                nn.Dropout(p=self.mc_dropout),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
                nn.Dropout(p=self.mc_dropout),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
            )

            self.fc = nn.Sequential(
                nn.Linear(in_features=16 * 5 * 5, out_features=120),
                nn.Dropout(p=self.mc_dropout),
                nn.ReLU(),
                nn.Linear(in_features=120, out_features=84),
                nn.Dropout(p=self.mc_dropout),
                nn.ReLU(),
                nn.Linear(in_features=84, out_features=self.cl_dim),
            )
        else:
            self.model = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.image_channels, out_channels=6, kernel_size=5
                ),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
            )

            self.fc = nn.Sequential(
                nn.Linear(in_features=16 * 5 * 5, out_features=120),
                nn.ReLU(),
                nn.Linear(in_features=120, out_features=84),
                nn.ReLU(),
                nn.Linear(in_features=84, out_features=self.cl_dim),
            )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return self.fc(out)
