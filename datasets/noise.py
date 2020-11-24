from functools import partial
from typing import OrderedDict

import torch
from torch.distributions import Bernoulli, Categorical, Normal, Uniform
from torch.utils.data import Dataset
import torchvision


class NoiseImageDataset(Dataset):
    def __init__(
        self,
        size,
        cl_dim=10,
        n_samples=1000,
        sample_type="uniform",
        target_type=("attr", "class"),
        transform=None,
        target_transform=None,
        **kwargs
    ):
        super().__init__()

        self.size = size
        self.cl_dim = cl_dim
        self.n_samples = n_samples
        self.sample_type = (sample_type,)
        self.target_type = target_type
        self.transform = transform
        self.target_transform = target_transform

        self.target_distributions = OrderedDict(
            {
                "attr": partial(Bernoulli(0.5).sample, (self.cl_dim,)),
                "class": lambda: Categorical(torch.full((self.cl_dim,), 1.0))
                .sample((1,))
                .view(-1)
                .item(),
            }
        )

        if sample_type == "uniform":
            self.distribution = Uniform(0.0, 1.0)
        elif sample_type == "normal":
            self.distribution = Normal(0.5, 1.0)
        else:
            raise ValueError("Unknown sample_type '{}'".format(sample_type))

    def __getitem__(self, index):
        img = self.distribution.sample(self.size)
        if self.sample_type == "normal":
            img.clamp_(0.0, 1.0)
        img = torchvision.transforms.functional.to_pil_image(img)

        target = []
        for t in self.target_type:
            sample = self.target_distributions[t]()
            if isinstance(sample, torch.Tensor):
                sample = sample.to(torch.long).view(-1)
            target.append(sample)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.n_samples


if __name__ == "__main__":
    dat = NoiseImageDataset((1, 28, 28), transform=torchvision.transforms.ToTensor())

    print(len(dat))

    x, y = dat[0]

    print("x shape: {}".format(x.shape))
    print("y shape: {}".format(y.shape))
