import logging
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, random_split

from config import Config
from datasets.datahandlers.cl import AwA2

EVAL_RATIO = 0.2


class ClAwA2(Dataset):
    def __init__(
        self,
        root: str = Config.clawa2_root,
        mode: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs.get("_log", None) is not None:
            self.log = kwargs["_log"].getChild("datasets.awa2")
        else:
            self.log = logging.getLogger("datasets.awa2")

        self.transform = transform
        self.target_transform = target_transform
        if mode not in ("train", "eval", "test"):
            raise ValueError(f"Unknown mode {mode}.")
        self.mode = mode

        self.awa2 = AwA2(
            root=root,
            split="test" if mode == "test" else "train",
            target_type=("attr", "class"),
        )
        self.n_classes = len(np.unique(self.awa2.img_index))

        self.compute_indices()

        self.class_index_to_trainclass_index = {
            c: i for i, c in enumerate(list(set(self.awa2.img_index)))
        }
        self.trainclass_to_class = {
            v: k for k, v in self.class_index_to_trainclass_index.items()
        }
        self.class_index_to_class_name = {
            k: self.awa2.class_index_to_class_name[self.trainclass_to_class[k]]
            for k in self.trainclass_to_class.keys()
        }

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
        x, (_, y) = self.awa2[self.indices[index]]
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
    dat = ClAwA2()
    print("Number of datapoints: {}".format(len(dat)))
