from typing import Tuple

import torch
import torch.nn as nn

LATENT_DIM = 100


class AutoEncoder(nn.Module):
    def __init__(
        self, image_channels: int = 3, latent_dim: int = LATENT_DIM, **kwargs
    ) -> None:
        super().__init__()

        if kwargs.get("image_size", 32) not in (32, 64):
            raise ValueError("This model does only work with 32x32 and 64x64 images!")

        self.encoder = Encoder(
            image_channels=image_channels, latent_dim=latent_dim, **kwargs
        )
        self.decoder = Decoder(
            image_channels=image_channels, latent_dim=latent_dim, **kwargs
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoding, loc, scale = self.encoder(x)
        reconstruction = self.decoder(encoding)

        return encoding, reconstruction, loc, scale

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        return self.decoder(code)


class Encoder(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        image_size: int = 32,
        latent_dim: int = LATENT_DIM,
        **kwargs,
    ) -> None:
        super().__init__()

        self.image_size = image_size

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

        feature_dim = self.features(
            torch.zeros(
                (
                    1,
                    image_channels,
                    self.image_size,
                    self.image_size,
                )
            )
        ).shape[1]

        self.fc_loc = nn.Sequential(
            nn.Linear(in_features=feature_dim, out_features=latent_dim)
        )

        self.fc_scale = nn.Sequential(
            nn.Linear(in_features=feature_dim, out_features=latent_dim),
            nn.Softplus(),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.features(x)

        loc = self.fc_loc(features)
        scale = self.fc_scale(features)
        return torch.randn_like(loc) * scale + loc, loc, scale


class Decoder(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        image_size: int = 32,
        latent_dim: int = LATENT_DIM,
        **kwargs,
    ) -> None:
        super().__init__()

        self.image_size = image_size

        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=256, kernel_size=7),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=5, stride=2
            ),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=image_channels if self.image_size == 32 else 128,
                kernel_size=5,
                stride=2,
                padding=1,
            ),
        )

        if self.image_size == 64:
            self.model.append(
                nn.ConvTranspose2d(
                    in_channels=128,
                    out_channels=image_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )

        self.model.append(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=image_channels,
                kernel_size=4,
                stride=1,
            )
        )

        if self.image_size == 64:
            self.model.append(
                nn.Conv2d(
                    in_channels=image_channels,
                    out_channels=image_channels,
                    kernel_size=4,
                    stride=1,
                )
            )

    def forward(self, code: torch.Tensor) -> torch.Tensor:
        code = code.reshape((code.shape[0], -1, 1, 1))
        return self.model(code)
