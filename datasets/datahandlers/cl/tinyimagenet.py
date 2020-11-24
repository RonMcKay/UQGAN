# Based on the following Gist:
# https://gist.github.com/z-a-f/b862013c0dc2b540cf96a123a6766e54

from collections import defaultdict
import logging
import os

from PIL import Image
import imageio
import numpy as np
from torch.utils.data import Dataset

dir_structure_help = r"""
TinyImageNetPath
├── test
│   └── images
│       ├── test_0.JPEG
│       ├── t...
│       └── ...
├── train
│   ├── n01443537
│   │   ├── images
│   │   │   ├── n01443537_0.JPEG
│   │   │   ├── n...
│   │   │   └── ...
│   │   └── n01443537_boxes.txt
│   ├── n01629819
│   │   ├── images
│   │   │   ├── n01629819_0.JPEG
│   │   │   ├── n...
│   │   │   └── ...
│   │   └── n01629819_boxes.txt
│   ├── n...
│   │   ├── images
│   │   │   ├── ...
│   │   │   └── ...
├── val
│   ├── images
│   │   ├── val_0.JPEG
│   │   ├── v...
│   │   └── ...
│   └── val_annotations.txt
├── wnids.txt
└── words.txt
"""


def download_and_unzip(URL, root_dir):
    error_message = "Download is not yet implemented. Please, go to {URL} yourself."
    raise NotImplementedError(error_message.format(URL))


def _add_channels(img, total_channels=3):
    while len(img.shape) < 3:  # third axis is the channels
        img = np.expand_dims(img, axis=-1)
    while (img.shape[-1]) < 3:
        img = np.concatenate([img, img[:, :, -1:]], axis=-1)
    return img


"""Creates a paths datastructure for the tiny imagenet.

Args:
  root_dir: Where the data is located
  download: Download if the data is not there

Members:
  label_id:
  ids:
  nit_to_words:
  data_dict:

"""


class TinyImageNetPaths:
    def __init__(self, root_dir, download=False):
        if download:
            download_and_unzip(
                "http://cs231n.stanford.edu/tiny-imagenet-200.zip", root_dir
            )
        train_path = os.path.join(root_dir, "train")
        val_path = os.path.join(root_dir, "val")
        test_path = os.path.join(root_dir, "test")

        wnids_path = os.path.join(root_dir, "wnids.txt")
        words_path = os.path.join(root_dir, "words.txt")

        self._make_paths(train_path, val_path, test_path, wnids_path, words_path)

    def _make_paths(self, train_path, val_path, test_path, wnids_path, words_path):
        self.ids = []
        with open(wnids_path, "r") as idf:
            for nid in idf:
                nid = nid.strip()
                self.ids.append(nid)
        self.nid_to_words = defaultdict(list)
        with open(words_path, "r") as wf:
            for line in wf:
                nid, labels = line.split("\t")
                labels = list(map(lambda x: x.strip(), labels.split(",")))
                self.nid_to_words[nid].extend(labels)

        self.paths = {
            "train": [],  # [img_path, id, nid, box]
            "val": [],  # [img_path, id, nid, box]
            "test": [],  # img_path
        }

        # Get the test paths
        self.paths["test"] = list(
            map(lambda x: os.path.join(test_path, x), os.listdir(test_path))
        )
        # Get the validation paths and labels
        with open(os.path.join(val_path, "val_annotations.txt")) as valf:
            for line in valf:
                fname, nid, x0, y0, x1, y1 = line.split()
                fname = os.path.join(val_path, "images", fname)
                bbox = int(x0), int(y0), int(x1), int(y1)
                label_id = self.ids.index(nid)
                self.paths["val"].append((fname, label_id, nid, bbox))

        # Get the training paths
        train_nids = os.listdir(train_path)
        for nid in train_nids:
            anno_path = os.path.join(train_path, nid, nid + "_boxes.txt")
            imgs_path = os.path.join(train_path, nid, "images")
            label_id = self.ids.index(nid)
            with open(anno_path, "r") as annof:
                for line in annof:
                    fname, x0, y0, x1, y1 = line.split()
                    fname = os.path.join(imgs_path, fname)
                    bbox = int(x0), int(y0), int(x1), int(y1)
                    self.paths["train"].append((fname, label_id, nid, bbox))

        # Be safe and always sort filenames
        self.paths["train"].sort(key=lambda x: x[0])
        self.paths["val"].sort(key=lambda x: x[0])
        self.paths["test"].sort(key=lambda x: x)


"""Datastructure for the tiny image dataset.

Args:
  root_dir: Root directory for the data
  mode: One of "train", "test", or "val"
  preload: Preload into memory
  load_transform: Transformation to use at the preload time
  transform: Transformation to use at the retrieval time
  download: Download the dataset

Members:
  tinp: Instance of the TinyImageNetPaths
  img_data: Image data
  label_data: Label data
"""


class TinyImageNet(Dataset):
    def __init__(
        self,
        root,
        split="train",
        preload=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        self.log = logging.getLogger("root.datahandlers.TinyImageNet")
        self.tinp = TinyImageNetPaths(root, download)
        self.mode = split
        self.label_idx = 1  # from [image, id, nid, box]
        self.preload = preload
        self.transform = transform
        self.target_transform = target_transform

        self.IMAGE_SHAPE = (64, 64, 3)

        self.img_data = []
        self.label_data = []

        self.samples = self.tinp.paths[split]
        self.samples_num = len(self.samples)

        if self.preload:
            self.log.info(f"Preloading {split} data...")
            self.img_data = np.zeros(
                (self.samples_num,) + self.IMAGE_SHAPE, dtype=np.float32
            )
            self.label_data = np.zeros((self.samples_num,), dtype=np.int)
            for idx in range(self.samples_num):
                s = self.samples[idx]
                img = imageio.imread(s[0])
                img = _add_channels(img)
                self.img_data[idx] = img
                if split != "test":
                    self.label_data[idx] = s[self.label_idx]

    def __len__(self):
        return self.samples_num

    def __getitem__(self, idx):
        if self.preload:
            img = self.img_data[idx]
            img = Image.fromarray(np.uint8(img))
            lbl = -1 if self.mode == "test" else self.label_data[idx]
        else:
            s = self.samples[idx]
            img = imageio.imread(s[0])
            lbl = -1 if self.mode == "test" else s[self.label_idx]

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            lbl = self.target_transform(lbl)

        return img, lbl


if __name__ == "__main__":
    dat = TinyImageNet("/data/datasets/cl/tinyimagenet")
