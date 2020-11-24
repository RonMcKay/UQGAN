import logging
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset
from torchvision.datasets import LSUN as torchLSUN

from config import Config


class LSUN(Dataset):
    categories = [
        "bedroom",
        "bridge",
        "church_outdoor",
        "classroom",
        "conference_room",
        "dining_room",
        "kitchen",
        "living_room",
        "restaurant",
        "tower",
    ]

    def __init__(
        self,
        root: str = Config.lsun_root,
        mode: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs.get("_log", None) is not None:
            self.log = kwargs["_log"].getChild("datasets.emnist")
        else:
            self.log = logging.getLogger("datasets.emnist")

        self.transform = transform
        self.target_transform = target_transform
        if mode not in ("train", "eval", "test"):
            raise ValueError(f"Unknown mode {mode}.")
        self.mode = mode

        self.dat = torchLSUN(root=root, classes={"eval": "val"}.get(mode, mode))
        self.n_classes = 10

        self.compute_indices()

        self.class_index_to_class_name = {
            i: self.categories[i] for i in range(len(self.categories))
        }

    def compute_indices(self):
        self.indices = list(range(len(self.dat)))

    def __getitem__(self, index):
        x, y = self.dat[self.indices[index]]
        y = torch.tensor(int(y), dtype=torch.long)

        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y

    def __len__(self):
        return len(self.indices)


if __name__ == "__main__":
    import torchvision.transforms as trans

    logging.basicConfig(level=logging.DEBUG)
    dat = LSUN(transform=trans.ToTensor())
    x, y = dat[0]
    dat.log.debug("Number of datapoints: {}".format(len(dat)))
