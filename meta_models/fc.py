import torch.nn as nn

from utils import init_weights


class MetaClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        act=nn.LeakyReLU(),
        **kwargs,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=256),
            act,
            nn.Linear(in_features=256, out_features=512),
            act,
            nn.Linear(in_features=512, out_features=1024),
            act,
            nn.Linear(in_features=1024, out_features=1),
            nn.Flatten(start_dim=0),
        )

        self.apply(init_weights)

    def forward(self, metrics):
        return self.model(metrics)
