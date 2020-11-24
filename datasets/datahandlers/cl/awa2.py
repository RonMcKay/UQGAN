# PyTorch Dataloader based on 'https://github.com/dfan/awa2-zero-shot-learning'

import logging
import os
from os.path import exists, isfile, join, splitext
from typing import Callable, Optional, Sequence, Union

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class AwA2(Dataset):
    def __init__(
        self,
        root: str = "/data/datasets/cl/Animals_with_Attributes2",
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        target_type: Union[Sequence[str], str] = ("attr", "class"),
    ):
        """

        Args:
            root: Root path of the dataset.
            split: Split to use. Valid options: ('train', 'test')
            transform: Image transforms.
            target_transform: Target transforms.
            target_type: Target types to return. Valid options: ('attr', 'class')
        """
        self.log = logging.getLogger(__name__ + ".AwA2")

        if not exists(root):
            raise ValueError("Root path '{}' does not exist!".format(root))
        self.root = root

        valid_split_types = ("train", "test")
        if split not in valid_split_types:
            raise ValueError(
                "Unknown split type '{}'. Valid split types: {}".format(
                    split, valid_split_types
                )
            )
        self.split = split

        valid_target_types = ("attr", "class")
        if not isinstance(target_type, (list, tuple)):
            target_type = (target_type,)
        for t in target_type:
            if t not in valid_target_types:
                raise ValueError(
                    "Unknown target type '{}'. Allowed types: {}".format(
                        t, ", ".join(valid_target_types)
                    )
                )
        self.target_type = target_type

        self.transform = transform
        self.target_transform = target_transform

        self.attributes_binary_matrix = np.array(
            np.genfromtxt(join(self.root, "predicate-matrix-binary.txt"), dtype="int")
        )
        self.attributes_binary_matrix = torch.from_numpy(self.attributes_binary_matrix)

        class_name_to_index = dict()
        # Build dictionary of indices to classes
        with open(join(self.root, "classes.txt")) as f:
            index = 0
            for line in f:
                class_name = line.split("\t")[1].strip()
                class_name_to_index[class_name] = index
                index += 1
        self.class_name_to_index = class_name_to_index
        self.class_index_to_class_name = {
            v: k for k, v in self.class_name_to_index.items()
        }
        self.class_index_to_attributes = {
            k: self.attributes_binary_matrix[k, :]
            for k in self.class_index_to_class_name.keys()
        }

        attribute_index_to_name = {}
        # Build dictionary of attribute indices to attribute names
        with open(join(self.root, "predicates.txt")) as f:
            for line in f:
                line = [i.strip() for i in line.split("\t")]
                line[0] = int(line[0]) - 1
                attribute_index_to_name[line[0]] = line[1]

        self.attribute_index_to_name = attribute_index_to_name
        self.attribute_name_to_index = {
            v: k for k, v in self.attribute_index_to_name.items()
        }

        img_names = []
        img_index = []
        with open(join(self.root, "{}classes.txt".format(self.split))) as f:
            for line in f:
                class_name = line.strip()
                folder_dir = join(self.root, "JPEGImages", class_name)

                class_index = class_name_to_index[class_name]
                for filename in os.listdir(folder_dir):
                    if (
                        isfile(join(folder_dir, filename))
                        and splitext(filename)[-1] == ".jpg"
                    ):
                        img_names.append(join(folder_dir, filename))
                        img_index.append(class_index)
        self.img_names = img_names
        self.img_index = img_index

    def __getitem__(self, index):
        img = Image.open(self.img_names[index])
        if img.getbands()[0] == "L":
            img = img.convert("RGB")

        img_class_index = self.img_index[index]

        target = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.class_index_to_attributes[img_class_index])
            elif t == "class":
                target.append(img_class_index)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.img_names)


if __name__ == "__main__":
    dat = AwA2()

    i = dat[0]
