import logging
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset, random_split

from config import Config
from datasets.datahandlers.cl import TinyImageNet as myTinyImageNet

EVAL_RATIO = 0.2


class TinyImageNet(Dataset):
    def __init__(
        self,
        root: str = Config.tinyimagenet_root,
        mode: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs.get("_log", None) is not None:
            self.log = kwargs["_log"].getChild("datasets.tinyimagenet")
        else:
            self.log = logging.getLogger("datasets.tinyimagenet")
        self.transform = transform
        self.target_transform = target_transform
        if mode not in ("train", "eval", "test"):
            raise ValueError(f"Unknown mode {mode}.")
        self.mode = mode

        self.dat = myTinyImageNet(
            root=root,
            split={"train": "train", "eval": "train", "test": "val"}.get(mode, "train"),
        )

        self.compute_indices()

        self.n_classes = 200
        self.class_index_to_class_name = {i: str(i) for i in range(self.n_classes)}

    def compute_indices(self):
        if self.mode == "test":
            self.indices = list(range(len(self.dat)))
            return

        t, e = random_split(
            range(len(self.dat)),
            [
                len(self.dat) - int(len(self.dat) * EVAL_RATIO),
                int(len(self.dat) * EVAL_RATIO),
            ],
            generator=torch.Generator().manual_seed(42),
        )
        if self.mode == "eval":
            self.indices = e.indices
        elif self.mode == "train":
            self.indices = t.indices

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
    logging.basicConfig(level=logging.DEBUG)
    dat = TinyImageNet()
    print("Number of datapoints: {}".format(len(dat)))
