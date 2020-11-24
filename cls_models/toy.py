import torch.nn as nn

from datasets.toy import ToyDataset

from .base import BaseClassifier


class ToyClassifier(BaseClassifier):
    def __init__(
        self,
        input_size: int = 2,
        cl_dim: int = ToyDataset.CL_DIM,
        **kwargs,
    ):
        super().__init__(cl_dim=cl_dim, **kwargs)

        self.act = nn.LeakyReLU()
        self.mc_dropout: float = kwargs.get("mc_dropout", 0.0)

        self.save_hyperparameters()

        if self.mc_dropout > 0:
            self.model = nn.Sequential(
                nn.Linear(in_features=input_size, out_features=512),
                nn.Dropout(p=self.mc_dropout),
                self.act,
                # nn.BatchNorm1d(num_features=512),
                nn.Linear(in_features=512, out_features=512),
                nn.Dropout(p=self.mc_dropout),
                self.act,
                # nn.BatchNorm1d(num_features=512),
                nn.Linear(in_features=512, out_features=512),
                nn.Dropout(p=self.mc_dropout),
                self.act,
                # nn.BatchNorm1d(num_features=512),
                nn.Linear(in_features=512, out_features=cl_dim),
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(in_features=input_size, out_features=512),
                self.act,
                # nn.BatchNorm1d(num_features=512),
                nn.Linear(in_features=512, out_features=512),
                self.act,
                # nn.BatchNorm1d(num_features=512),
                nn.Linear(in_features=512, out_features=512),
                self.act,
                # nn.BatchNorm1d(num_features=512),
                nn.Linear(in_features=512, out_features=cl_dim),
            )

    def _forward(self, x):
        return self.model(x)
