from pytorch_lightning import LightningModule
import torch
from torch.distributions.uniform import Uniform
import torch.nn as nn

from datasets.toy import ToyDataset
from utils import init_gan_weights


class ToyGenerator(LightningModule):
    def __init__(
        self,
        output_size: int = 2,
        cl_dim: int = ToyDataset.CL_DIM,
        latent_dim: int = 100,
        conditional=True,
        **kwargs,
    ):
        super().__init__()

        if conditional and cl_dim is None:
            raise ValueError(
                "When conditional is set to 'true', 'cl_dim' must be specified."
            )

        self.latent_dim = latent_dim
        self.latent_distribution = Uniform(0, 1)
        self.conditional = conditional
        self.act = nn.ReLU()

        if self.conditional:
            self.model = nn.Sequential(
                nn.Linear(in_features=cl_dim + self.latent_dim, out_features=1024),
                self.act,
                nn.BatchNorm1d(num_features=1024),
                nn.Linear(in_features=1024, out_features=512),
                self.act,
                nn.BatchNorm1d(num_features=512),
                nn.Linear(in_features=512, out_features=256),
                self.act,
                nn.BatchNorm1d(num_features=256),
                nn.Linear(in_features=256, out_features=output_size),
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(in_features=self.latent_dim, out_features=1024),
                self.act,
                nn.BatchNorm1d(num_features=1024),
                nn.Linear(in_features=1024, out_features=512),
                self.act,
                nn.BatchNorm1d(num_features=512),
                nn.Linear(in_features=512, out_features=256),
                self.act,
                nn.BatchNorm1d(num_features=256),
                nn.Linear(in_features=256, out_features=output_size),
            )

        self.apply(init_gan_weights)

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

        return self.model(torch.cat((cl, latent_code), 1))

    def _forward(self, num_samples=None, z=None):
        if z is None:
            if num_samples is None:
                raise ValueError("Either 'num_samples' or 'z' should be specified.")
            latent_code = self.latent_distribution.sample(
                (num_samples, self.latent_dim)
            ).to(next(self.parameters()).device)
        else:
            latent_code = z

        return self.model(latent_code)


class ToyDiscriminator(LightningModule):
    def __init__(
        self,
        input_size: int = 2,
        cl_dim: int = ToyDataset.CL_DIM,
        conditional: bool = True,
        **kwargs,
    ):
        super().__init__()

        if conditional and cl_dim is None:
            raise ValueError(
                "When conditional is set to 'true', 'cl_dim' must be specified."
            )

        self.conditional = conditional
        self.act = nn.LeakyReLU()

        if self.conditional:
            self.model = nn.Sequential(
                nn.Linear(in_features=input_size + cl_dim, out_features=512),
                self.act,
                nn.Linear(in_features=512, out_features=512),
                self.act,
                nn.Linear(in_features=512, out_features=512),
                self.act,
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(in_features=input_size, out_features=512),
                self.act,
                nn.Linear(in_features=512, out_features=512),
                self.act,
                nn.Linear(in_features=512, out_features=512),
                self.act,
            )

        self.fc = nn.Linear(in_features=512, out_features=1)

        self.apply(init_gan_weights)

        self.save_hyperparameters()

    def forward(self, *args, **kwargs):
        if self.conditional:
            return self._forward_conditional(*args, **kwargs)
        else:
            return self._forward(*args, **kwargs)

    def _forward_conditional(self, x, cl):
        x = torch.cat((x, cl), 1)

        out = self.model(x)
        out_dc = self.fc(out)

        return out_dc.view(-1)

    def _forward(self, x):
        out = self.model(x)
        out_dc = self.fc(out)

        return out_dc.view(-1)


class LargerToyGenerator(nn.Module):
    def __init__(
        self,
        output_size: int = 2,
        cl_dim: int = ToyDataset.CL_DIM,
        latent_dim: int = 100,
        conditional=True,
        **kwargs,
    ):
        super().__init__()

        if conditional and cl_dim is None:
            raise ValueError(
                "When conditional is set to 'true', 'cl_dim' must be specified."
            )

        self.latent_dim = latent_dim
        self.latent_distribution = Uniform(0, 1)
        self.conditional = conditional
        self.act = nn.ReLU()

        if self.conditional:
            self.model = nn.Sequential(
                nn.Linear(in_features=cl_dim + self.latent_dim, out_features=2048),
                self.act,
                nn.BatchNorm1d(num_features=2048),
                nn.Linear(in_features=2048, out_features=1024),
                self.act,
                nn.BatchNorm1d(num_features=1024),
                nn.Linear(in_features=1024, out_features=512),
                self.act,
                nn.BatchNorm1d(num_features=512),
                nn.Linear(in_features=512, out_features=output_size),
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(in_features=self.latent_dim, out_features=2048),
                self.act,
                nn.BatchNorm1d(num_features=2048),
                nn.Linear(in_features=2048, out_features=1024),
                self.act,
                nn.BatchNorm1d(num_features=1024),
                nn.Linear(in_features=1024, out_features=512),
                self.act,
                nn.BatchNorm1d(num_features=512),
                nn.Linear(in_features=512, out_features=output_size),
            )

        self.apply(init_gan_weights)

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

        return self.model(torch.cat((cl, latent_code), 1))

    def _forward(self, num_samples=None, z=None):
        if z is None:
            if num_samples is None:
                raise ValueError("Either 'num_samples' or 'z' should be specified.")
            latent_code = self.latent_distribution.sample(
                (num_samples, self.latent_dim)
            ).to(next(self.parameters()).device)
        else:
            latent_code = z

        return self.model(latent_code)


class LargerToyDiscriminator(nn.Module):
    def __init__(
        self,
        input_size: int = 2,
        cl_dim: int = ToyDataset.CL_DIM,
        conditional: bool = True,
        **kwargs,
    ):
        super().__init__()

        if conditional and cl_dim is None:
            raise ValueError(
                "When conditional is set to 'true', 'cl_dim' must be specified."
            )

        self.conditional = conditional
        self.act = nn.LeakyReLU()

        if self.conditional:
            self.model = nn.Sequential(
                nn.Linear(in_features=input_size + cl_dim, out_features=1024),
                self.act,
                nn.Linear(in_features=1024, out_features=1024),
                self.act,
                nn.Linear(in_features=1024, out_features=1024),
                self.act,
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(in_features=input_size, out_features=1024),
                self.act,
                nn.Linear(in_features=1024, out_features=1024),
                self.act,
                nn.Linear(in_features=1024, out_features=1024),
                self.act,
            )

        self.fc = nn.Linear(in_features=1024, out_features=1)

        self.apply(init_gan_weights)

    def forward(self, *args, **kwargs):
        if self.conditional:
            return self._forward_conditional(*args, **kwargs)
        else:
            return self._forward(*args, **kwargs)

    def _forward_conditional(self, x, cl):
        x = torch.cat((x, cl), 1)

        out = self.model(x)
        out_dc = self.fc(out)

        return out_dc.view(-1)

    def _forward(self, x):
        out = self.model(x)
        out_dc = self.fc(out)

        return out_dc.view(-1)


if __name__ == "__main__":
    from nnaddons import count_parameters

    N_SAMPLES = 3
    CL_DIM = ToyDataset.CL_DIM
    # CL_DIM = 10

    LATENT_DIM = 10

    gen = ToyGenerator(cl_dim=CL_DIM, latent_dim=LATENT_DIM)
    x_tilde = gen(torch.zeros(N_SAMPLES, CL_DIM))
    print("Generator output shape: {}".format(x_tilde.shape))
    print("Generator parameters: {}".format(count_parameters(gen)))

    disc = ToyDiscriminator(cl_dim=CL_DIM)
    disc_x_tilde, _ = disc(x_tilde, torch.zeros((N_SAMPLES, CL_DIM)))
    print("Discriminator output shape: {}".format(disc_x_tilde.shape))
    print("Discriminator parameters: {}".format(count_parameters(disc)))
