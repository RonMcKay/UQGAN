from typing import Sequence, Tuple, Union

from sklearn.datasets import make_moons
import torch
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
import torch.nn.functional as tf
from torch.utils.data import Dataset


class ToyDataset(Dataset):
    N_SAMPLES = 50000

    means = [
        torch.tensor((4.0, 0.0)),
        torch.tensor((-4.0, 0.0)),
    ]
    std = torch.tensor(0.4)
    CL_DIM = 2

    def __init__(
        self, n_samples: int = N_SAMPLES, target_transform=None, **kwargs
    ) -> None:
        super().__init__()

        self.n_samples = n_samples
        self.target_transform = target_transform

        self.data = []
        for m in self.means[:-1]:
            self.data.append(
                Normal(loc=m, scale=self.std).sample(
                    (self.n_samples // len(self.means),)
                )
            )
        self.data.append(
            Normal(loc=self.means[-1], scale=self.std).sample(
                (
                    self.n_samples
                    - (len(self.means) - 1) * (self.n_samples // len(self.means)),
                )
            )
        )

        self.targets = []
        for i, d in enumerate(self.data):
            self.targets.append(torch.full((d.shape[0],), i))

        self.data = torch.cat(self.data)
        self.targets = torch.cat(self.targets).type(torch.LongTensor)

    def __getitem__(
        self, index
    ) -> Tuple[torch.Tensor, Sequence[Union[torch.Tensor, int]]]:
        target = self.targets[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return self.data[index], target

    def __len__(self) -> int:
        return self.n_samples


class ToyDataset2(Dataset):
    N_SAMPLES = 50000

    means = [
        torch.tensor((2.0, 0.0)),
        torch.tensor((-2.0, 0.0)),
    ]
    std = torch.tensor(1.0)
    CL_DIM = 2

    def __init__(
        self, n_samples: int = N_SAMPLES, target_transform=None, **kwargs
    ) -> None:
        super().__init__()

        self.n_samples = n_samples
        self.target_transform = target_transform

        self.data = torch.empty((0,))
        self.targets = torch.empty((0,))
        for i, m in enumerate(self.means[:-1]):
            n = self.n_samples // len(self.means)
            self.data = torch.cat(
                (self.data, Normal(loc=m, scale=self.std).sample((n,)))
            )
            self.targets = torch.cat((self.targets, torch.full((n,), i)))

        n = self.n_samples - (len(self.means) - 1) * (self.n_samples // len(self.means))
        self.data = torch.cat(
            (self.data, Normal(loc=self.means[-1], scale=self.std).sample((n,)))
        )
        self.targets = torch.cat((self.targets, torch.full((n,), i + 1))).type(
            torch.LongTensor
        )

    def __getitem__(
        self, index
    ) -> Tuple[torch.Tensor, Sequence[Union[torch.Tensor, int]]]:
        target = self.targets[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return self.data[index], target

    def __len__(self) -> int:
        return self.n_samples


class ToyDataset3(Dataset):
    N_SAMPLES = 50000

    CL_DIM = 2

    def __init__(
        self, n_samples: int = N_SAMPLES, target_transform=None, noise=0.1, **kwargs
    ) -> None:
        super().__init__()

        self.n_samples = n_samples
        self.target_transform = target_transform

        self.data, self.targets = make_moons(
            n_samples=self.n_samples, shuffle=False, noise=noise
        )
        self.data = torch.from_numpy(self.data).to(torch.float)
        self.targets = torch.from_numpy(self.targets).type(torch.LongTensor)

    def __getitem__(
        self, index
    ) -> Tuple[torch.Tensor, Sequence[Union[torch.Tensor, int]]]:
        target = self.targets[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return self.data[index], target

    def __len__(self) -> int:
        return self.n_samples


class ToyDataset4(Dataset):
    N_SAMPLES = 50000

    means = []
    for i in range(3):
        for j in range(3):
            means.append(torch.tensor((i * 4.0, j * 4.0)))
    std = torch.tensor(1.0)

    CL_DIM = 9

    def __init__(
        self, n_samples: int = N_SAMPLES, target_transform=None, **kwargs
    ) -> None:
        super().__init__()

        self.n_samples = n_samples
        self.target_transform = target_transform

        self.data = torch.empty((0,))
        self.targets = torch.empty((0,))
        for i, m in enumerate(self.means[:-1]):
            n = self.n_samples // len(self.means)
            self.data = torch.cat(
                (self.data, Normal(loc=m, scale=self.std).sample((n,)))
            )
            self.targets = torch.cat((self.targets, torch.full((n,), i)))

        n = self.n_samples - (len(self.means) - 1) * (self.n_samples // len(self.means))
        self.data = torch.cat(
            (self.data, Normal(loc=self.means[-1], scale=self.std).sample((n,)))
        )
        self.targets = torch.cat((self.targets, torch.full((n,), i + 1))).type(
            torch.LongTensor
        )
        self.y_onehot = tf.one_hot(self.targets, num_classes=self.CL_DIM)

    def __getitem__(
        self, index
    ) -> Tuple[torch.Tensor, Sequence[Union[torch.Tensor, int]]]:
        target = self.targets[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return self.data[index], target

    def __len__(self) -> int:
        return self.n_samples


class ToyDataset5(Dataset):
    N_SAMPLES = 50000

    means = []
    for i in range(3):
        for j in range(3):
            means.append(torch.tensor((i * 4.0, j * 4.0)))
    std = torch.tensor(1.0)

    classes = [0, 1, 2, 1, 2, 0, 2, 0, 1]

    CL_DIM = 3

    def __init__(
        self, n_samples: int = N_SAMPLES, target_transform=None, **kwargs
    ) -> None:
        super().__init__()

        self.n_samples = n_samples
        self.target_transform = target_transform

        self.data = torch.empty((0,))
        self.targets = torch.empty((0,))
        for i, m in zip(self.classes[:-1], self.means[:-1]):
            n = self.n_samples // len(self.means)
            self.data = torch.cat(
                (self.data, Normal(loc=m, scale=self.std).sample((n,)))
            )
            self.targets = torch.cat((self.targets, torch.full((n,), i)))

        n = self.n_samples - (len(self.means) - 1) * (self.n_samples // len(self.means))
        self.data = torch.cat(
            (self.data, Normal(loc=self.means[-1], scale=self.std).sample((n,)))
        )
        self.targets = torch.cat(
            (self.targets, torch.full((n,), self.classes[-1]))
        ).type(torch.LongTensor)

        self.y_onehot = tf.one_hot(self.targets, num_classes=self.CL_DIM)

    def __getitem__(
        self, index
    ) -> Tuple[torch.Tensor, Sequence[Union[torch.Tensor, int]]]:
        target = self.targets[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return self.data[index], target

    def __len__(self) -> int:
        return self.n_samples


class ToyOODDataset(Dataset):
    N_SAMPLES = 50000

    def __init__(
        self,
        n_samples: int = N_SAMPLES,
        target_transform=None,
        low=None,
        high=None,
        **kwargs
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.target_transform = target_transform
        self.in_means = torch.stack(ToyDataset.means)

        self.data = Uniform(
            low=-5 if low is None else low, high=5 if high is None else high
        ).sample((2 * self.n_samples,))
        self.data = self.data.view((self.n_samples, 2))

        dists = torch.cdist(self.data, self.in_means, p=2)
        dists = dists.min(dim=1)[0]
        self.data = self.data[dists > (ToyDataset.std * 2 / ToyDataset.std)]
        self.targets = torch.full((self.data.shape[0],), -1.0).type(torch.LongTensor)

    def __getitem__(
        self, index
    ) -> Tuple[torch.Tensor, Sequence[Union[torch.Tensor, int]]]:
        target = self.targets[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return self.data[index], target

    def __len__(self) -> int:
        return self.data.shape[0]


if __name__ == "__main__":
    import matplotlib  # noqa: F401

    # matplotlib.use("Qt5Agg")
    import matplotlib.pyplot as plt

    dat = ToyDataset4(n_samples=5000)
    cm = plt.get_cmap("tab10")
    c = torch.unique(dat.targets, return_inverse=True)[1]
    # c = (c - c.min()) / (c.max() - c.min())
    cols = cm(c)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(
        dat.data[:, 0],
        dat.data[:, 1],
        c=cols,
        marker="o",
        alpha=0.5,
        s=4,
    )
    plt.show(block=True)
