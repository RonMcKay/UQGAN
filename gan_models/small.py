from pytorch_lightning import LightningModule
import torch
from torch.distributions.uniform import Uniform
import torch.nn as nn

from utils import init_weights

LATENT_DIM = 128
N_FEATUREMAPS = 128


class SmallGenerator(LightningModule):
    def __init__(
        self,
        cl_dim=None,
        n_featuremaps=N_FEATUREMAPS,
        latent_dim=LATENT_DIM,
        image_channels=3,
        conditional=True,
        **kwargs
    ):
        super().__init__()

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
            self.fc = nn.Sequential(
                nn.Linear(cl_dim + self.latent_dim, 4 * 4 * 4 * self.n_featuremaps),
                nn.BatchNorm1d(4 * 4 * 4 * self.n_featuremaps),
                self.act,
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(self.latent_dim, 4 * 4 * 4 * self.n_featuremaps),
                nn.BatchNorm1d(4 * 4 * 4 * self.n_featuremaps),
                self.act,
            )

        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                4 * self.n_featuremaps,
                2 * self.n_featuremaps,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            nn.BatchNorm2d(2 * self.n_featuremaps),
            self.act,
            nn.ConvTranspose2d(
                2 * self.n_featuremaps,
                self.n_featuremaps,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            nn.BatchNorm2d(self.n_featuremaps),
            self.act,
            nn.ConvTranspose2d(
                self.n_featuremaps,
                self.image_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
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
        out = self.model(out.reshape(cl.shape[0], 4 * self.n_featuremaps, 4, 4))

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
                4 * self.n_featuremaps,
                4,
                4,
            )
        )

        return out


class SmallDiscriminator(LightningModule):
    def __init__(
        self,
        cl_dim,
        n_featuremaps=N_FEATUREMAPS,
        image_channels=3,
        conditional=True,
        **kwargs
    ):
        super().__init__()

        if conditional and cl_dim is None:
            raise ValueError(
                "When conditional is set to 'true', 'cl_dim' must be specified."
            )

        self.n_featuremaps = n_featuremaps
        self.conditional = conditional
        self.act = nn.LeakyReLU()

        if self.conditional:
            self.model = nn.Sequential(
                nn.Conv2d(
                    image_channels + cl_dim if not self.acgan else image_channels,
                    self.n_featuremaps,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                ),
                self.act,
                nn.Conv2d(
                    self.n_featuremaps,
                    2 * self.n_featuremaps,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                ),
                self.act,
                nn.Conv2d(
                    2 * self.n_featuremaps,
                    4 * self.n_featuremaps,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                ),
                self.act,
            )
        else:
            self.model = nn.Sequential(
                nn.Conv2d(
                    image_channels,
                    self.n_featuremaps,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                ),
                self.act,
                nn.Conv2d(
                    self.n_featuremaps,
                    2 * self.n_featuremaps,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                ),
                self.act,
                nn.Conv2d(
                    2 * self.n_featuremaps,
                    4 * self.n_featuremaps,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                ),
                self.act,
            )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 4 * 4 * self.n_featuremaps, 1),
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
        cl = cl.expand(list(cl.shape[:2]) + [x.shape[2], x.shape[3]])
        x = torch.cat((x, cl), 1)

        x = self.model(x)
        out_dc = self.fc(x)

        return out_dc

    def _forward(self, x):
        x = self.model(x)
        out_dc = self.fc(x)

        return out_dc
