from pytorch_lightning import LightningModule
import torch
from torch.distributions.normal import Normal
import torch.nn as nn

LATENT_DIM = 100


class Generator(LightningModule):
    def __init__(self, latent_dim=LATENT_DIM, **kwargs) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.latent_distribution = Normal(0, 1)

        self.model = nn.Sequential(
            nn.Linear(latent_dim + 2, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, latent_dim),
            nn.Softplus(),
        )

        self.save_hyperparameters()

    def forward(self, code):
        code = torch.cat(
            (torch.randn((code.shape[0], 2), device=code.device), code), dim=1
        )
        return self.model(code)


class DiscriminatorLatentSpace(LightningModule):
    def __init__(self, latent_dim=LATENT_DIM, **kwargs) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
            nn.Flatten(start_dim=0),
        )

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)


class DiscriminatorImageSpace(LightningModule):
    def __init__(self, image_channels=3, image_size=32, **kwargs) -> None:
        super().__init__()

        self.image_channels = image_channels
        self.image_size = image_size

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=self.image_channels, out_channels=20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

        feature_dim = self.features(
            torch.zeros((1, self.image_channels, self.image_size, self.image_size))
        ).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(in_features=feature_dim, out_features=500),
            nn.ReLU(),
            nn.Linear(500, 1),
            nn.Flatten(start_dim=0),
        )

        self.save_hyperparameters()

    def forward(self, x):
        features = self.features(x)
        return self.fc(features)
