# Standard Library
from collections import defaultdict
import logging
from os.path import exists, join
from typing import Callable, List, Optional, Sequence, Union

# Thirdparty libraries
from PIL import Image
import torch
from torch.utils.data import Dataset


class CUB200(Dataset):
    def __init__(
        self,
        root: str = "/data/datasets/cl/CUB_200_2011",
        split: str = "train",
        minimum_attribute_certainty: int = 2,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        target_type: Union[Sequence[str], str] = ("attr", "class"),
        **kwargs
    ):
        """

        Args:
            root: Root path of the dataset.
            split: Split to use. Valid options: ('train', 'test', 'all')
            minimum_attribute_certainty: Minimum certainty of the annotated attribute
                as provided by the human annotator. Valid options: (1, 2, 3, 4)
            transform: Image transforms.
            target_transform: Target transforms.
            target_type: Target types to return. Valid options: ('attr', 'class')
        """
        self.log = logging.getLogger(__name__ + ".CUB200")

        if not exists(root):
            self.log.error("Root path '{}' does not exist!".format(root))
            raise ValueError("Root path '{}' does not exist!".format(root))
        self.root = root

        valid_split_types = ("train", "test", "all")
        if split not in valid_split_types:
            raise ValueError(
                "Unknown split type '{}'. Valid split types: {}".format(
                    split,
                    ", ".join(valid_split_types),
                )
            )
        self.split = split

        valid_attribute_certainties = (1, 2, 3, 4)
        if minimum_attribute_certainty not in valid_attribute_certainties:
            raise ValueError(
                "Unknown attribute certainty '{}'. Valid certainties: {}".format(
                    minimum_attribute_certainty,
                    ", ".join(valid_attribute_certainties),
                )
            )
        self.minimum_attribute_certainty = minimum_attribute_certainty

        valid_target_types = ("attr", "class")
        for t in target_type:
            if t not in valid_target_types:
                raise ValueError(
                    "Unknown target type '{}'. Allowed types: {}".format(
                        t,
                        ", ".join(valid_target_types),
                    )
                )
        self.target_type = target_type

        self.transform = transform
        self.target_transform = target_transform

        # initialize variables for loading
        self.image_id_to_image_file = {}
        self.split_ids = []
        self.class_index_to_class_name = {}
        self.image_index_to_class_index = {}
        self.attribute_index_to_attribute_name = {}
        self.image_index_to_attribute_indices = defaultdict(list)

        # load image filenames
        with open(join(self.root, "images.txt"), "r") as f:
            line = f.readline()
            while line:
                line = line.strip().split(" ")
                self.image_id_to_image_file[int(line[0]) - 1] = join(
                    self.root, "images", line[1]
                )
                line = f.readline()

        self.image_file_to_id = {v: k for k, v in self.image_id_to_image_file.items()}

        # get image IDs for split
        if split == "train":
            self.load_split("1")
        elif split == "test":
            self.load_split("0")
        elif split == "all":
            self.split_ids = list(self.image_id_to_image_file.keys())

        # get class names
        with open(join(self.root, "classes.txt"), "r") as f:
            line = f.readline()
            while line:
                line = line.strip().split(" ")
                self.class_index_to_class_name[int(line[0]) - 1] = line[1]
                line = f.readline()

        self.class_name_to_class_index = {
            v: k for k, v in self.class_index_to_class_name.items()
        }

        # get image class labels
        with open(join(self.root, "image_class_labels.txt"), "r") as f:
            line = f.readline()
            while line:
                line = line.strip().split(" ")
                self.image_index_to_class_index[int(line[0]) - 1] = int(line[1]) - 1
                line = f.readline()

        # get attribute names
        with open(join(self.root, "attributes", "attributes.txt"), "r") as f:
            line = f.readline()
            while line:
                line = line.strip().split(" ")
                self.attribute_index_to_attribute_name[int(line[0]) - 1] = line[1]
                line = f.readline()

        self.attribute_name_to_attribute_index = {
            v: k for k, v in self.attribute_index_to_attribute_name.items()
        }
        self.n_attributes = len(self.attribute_index_to_attribute_name)

        # get image attributes
        with open(
            join(self.root, "attributes", "image_attribute_labels.txt"), "r"
        ) as f:
            line = f.readline()
            while line:
                line = line.strip().split(" ")
                if line[2] == "1" and int(line[3]) >= self.minimum_attribute_certainty:
                    self.image_index_to_attribute_indices[int(line[0]) - 1].append(
                        int(line[1]) - 1
                    )
                line = f.readline()

    def load_split(self, split_id) -> None:
        with open(join(self.root, "train_test_split.txt"), "r") as f:
            line = f.readline()
            while line:
                line = line.strip().split(" ")
                if line[1] == split_id:
                    self.split_ids.append(int(line[0]) - 1)
                line = f.readline()

    def __getitem__(self, index):
        img = Image.open(self.image_id_to_image_file[self.split_ids[index]])
        if img.getbands()[0] == "L":
            img = img.convert("RGB")

        img_class_index = self.image_index_to_class_index[self.split_ids[index]]

        target = []
        for t in self.target_type:
            if t == "attr":
                target.append(
                    self.img_attributes_to_one_hot(
                        self.image_index_to_attribute_indices[self.split_ids[index]]
                    )
                )
            elif t == "class":
                target.append(img_class_index)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def img_attributes_to_one_hot(self, img_attributes: List[int]) -> torch.Tensor:
        one_hot = torch.LongTensor(self.n_attributes).zero_()
        if len(img_attributes) > 0:
            one_hot.scatter_(0, torch.tensor(img_attributes), 1)

        return one_hot

    def __len__(self):
        return len(self.split_ids)


if __name__ == "__main__":
    # Thirdparty libraries
    from torch.utils.data import DataLoader
    import torchvision.transforms as trans

    train_dat = CUB200(
        transform=trans.Compose(
            [trans.Resize(128), trans.CenterCrop(128), trans.ToTensor()]
        ),
        split="train",
    )
    test_dat = CUB200(split="test")
    all_dat = CUB200(split="all")

    print("Training images: {}".format(len(train_dat)))
    print("Testing images: {}".format(len(test_dat)))
    print("Total number of images: {}".format(len(all_dat)))

    datloader = DataLoader(train_dat, batch_size=64)
    idatloader = iter(datloader)

    x, (att, y) = idatloader.next()
    print("x shape: {}".format(x.shape))
    print("att shape: {}".format(att.shape))
