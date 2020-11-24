from pytorch_lightning import LightningModule
import torch
from torch.distributions.normal import Normal
import torch.nn as nn

from utils import init_weights

LATENT_DIM = 128


class Generator(LightningModule):
    def __init__(
        self,
        cl_dim=None,
        latent_dim=LATENT_DIM,
        image_channels=3,
        conditional=True,
        **kwargs
    ):
        super().__init__()

        if "image_size" in kwargs and kwargs["image_size"] not in (32, 64):
            raise ValueError(
                "DCGAN is only usable with an image size of 32x32 or 64x64."
            )

        self.latent_dim = latent_dim
        self.latent_distribution = Normal(0, 1)
        self.conditional = conditional

        if conditional and cl_dim is None:
            raise ValueError(
                "When conditional is set to 'true', 'cl_dim' must be specified."
            )

        def block(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False
        ):
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )

        if self.conditional:
            self.fc_embedding = nn.Linear(cl_dim + self.latent_dim, 4 * 4 * 1024)
        else:
            self.fc_embedding = nn.Linear(self.latent_dim, 4 * 4 * 1024)

        self.layer1 = block(1024, 512)
        self.layer2 = block(512, 256)
        self.layer3 = block(256, 128)
        if kwargs.get("image_size", 32) == 64:
            self.layer3 = nn.Sequential(self.layer3, block(128, 128))
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(
                128, image_channels, kernel_size=3, stride=1, padding=1, bias=False
            ),
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

        out = self.fc_embedding(torch.cat((latent_code, cl), 1))
        out = out.reshape((cl.shape[0], 1024, 4, 4))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

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

        out = self.fc_embedding(latent_code)
        out = out.reshape(
            (z.shape[0] if num_samples is None else num_samples, 1024, 4, 4)
        )
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out


class Discriminator(LightningModule):
    def __init__(self, cl_dim=None, image_channels=3, conditional=True, **kwargs):
        super().__init__()

        self.conditional = conditional

        if conditional and cl_dim is None:
            raise ValueError(
                "When conditional is set to 'true', 'cl_dim' must be specified."
            )

        def block(
            in_channels,
            out_channels,
            input_spatial_size,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        ):
            return nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                nn.LayerNorm([out_channels] + input_spatial_size),
                nn.LeakyReLU(0.2, inplace=True),
            )

        if kwargs.get("image_size", 32) == 64:
            scale_factor = 2
        else:
            scale_factor = 1

        if self.conditional:
            self.layer1 = nn.Sequential(
                nn.Conv2d(
                    image_channels + cl_dim,
                    128 // scale_factor,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.layer1 = nn.Sequential(
                nn.Conv2d(
                    image_channels,
                    128 // scale_factor,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.layer2 = block(
            128 // scale_factor,
            256 // scale_factor,
            [16 * scale_factor, 16 * scale_factor],
        )
        self.layer3 = block(
            256 // scale_factor,
            512 // scale_factor,
            [8 * scale_factor, 8 * scale_factor],
        )
        self.layer4 = block(
            512 // scale_factor,
            1024 // scale_factor,
            [4 * scale_factor, 4 * scale_factor],
        )
        if kwargs.get("image_size", 32) == 64:
            self.layer4 = nn.Sequential(self.layer4, block(512, 1024, [4, 4]))
        self.layer5 = nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0, bias=False)

        self.apply(init_weights)

        self.save_hyperparameters()

    def forward(self, *args, **kwargs):
        if self.conditional:
            return self._forward_conditional(*args, **kwargs)
        else:
            return self._forward(*args, **kwargs)

    def _forward_conditional(self, x, cl):
        cl_stack = cl.unsqueeze(-1).unsqueeze(-1)
        cl_stack = cl_stack.expand(list(cl_stack.shape)[:2] + [x.shape[2], x.shape[3]])

        out = self.layer1(torch.cat((x, cl_stack), 1))
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        return out.reshape(out.shape[0], -1)

    def _forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        return out.reshape(out.shape[0], -1)


if __name__ == "__main__":
    # this is for debugging purposes
    cl_dim = 10
    n_samples = 1
    print("'cl_dim': {} || 'n_samples': {}".format(cl_dim, n_samples))

    gen = Generator(cl_dim)
    x_tilde = gen(torch.zeros((n_samples, cl_dim)))
    print("Generator output shape: {}".format(x_tilde.shape))

    disc = Discriminator(cl_dim)
    disc_x_tilde = disc(x_tilde, torch.zeros((n_samples, cl_dim)))
    print("Discriminator output shape: {}".format(disc_x_tilde.shape))
