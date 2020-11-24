import logging
from typing import List

import torch
from torch.utils.data import Dataset


class ClassSubset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        class_list: List[int],
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs.get("_log", None) is not None:
            self.log = kwargs["_log"].getChild("datasets.class_subset")
        else:
            self.log = logging.getLogger("datasets.class_subset")
        self.dataset = dataset
        self.classes = class_list
        self.class_old_to_new = {c: i for i, c in enumerate(self.classes)}
        self.class_new_to_old = {v: k for k, v in self.class_old_to_new.items()}

        self.transform = getattr(self.dataset, "transform", None)
        if self.transform is not None:
            self.dataset.transform = None

        self.target_transform = getattr(self.dataset, "target_transform", None)
        if self.target_transform is not None:
            self.dataset.target_transform = None

        self.indices = []
        self.filter_classes()

        self._class_index_to_class_name = getattr(
            self.dataset, "class_index_to_class_name", None
        )
        self.class_index_to_class_name = None
        if self._class_index_to_class_name is not None:
            self.class_index_to_class_name = dict()
            for k, v in self._class_index_to_class_name.items():
                if k in self.class_old_to_new:
                    self.class_index_to_class_name[self.class_old_to_new[k]] = v

    def filter_classes(self):
        self.log.debug("Start class filtering")
        for i in range(len(self.dataset)):
            _, y = self.dataset[i]
            if isinstance(y, tuple) and len(y) == 2:
                y = y[1]
            if y in self.classes:
                self.indices.append(i)

    def __getitem__(self, index):
        x, y = self.dataset[self.indices[index]]
        y = torch.tensor(self.class_old_to_new[int(y)]).type(torch.LongTensor)

        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y

    def __len__(self):
        return len(self.indices)
