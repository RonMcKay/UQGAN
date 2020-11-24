import logging
from typing import Callable, Optional, Sequence, Union

from torch.utils.data import Dataset
from torchvision.datasets import CelebA as torchCelebA

from config import Config


class CelebA(Dataset):
    def __init__(
        self,
        root: str = Config.celeba_root,
        mode: str = "train",
        target_type: Union[Sequence[str], str] = ["attr", "identity"],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        **kwargs,
    ):
        super().__init__()
        if kwargs.get("_log", None) is not None:
            self.log = kwargs["_log"].getChild("datasets.celeba")
        else:
            self.log = logging.getLogger("datasets.celeba")

        if isinstance(target_type, str):
            target_type = [target_type]

        self.target_type = target_type
        self.transform = transform
        self.target_transform = target_transform
        if mode not in ("train", "eval", "test"):
            raise ValueError(f"Unknown mode {mode}.")
        self.mode = mode

        self.dat = torchCelebA(
            root=root,
            split={"eval": "valid"}.get(mode, mode),
            target_type=self.target_type,
            download=download,
        )
        self.class_index_to_class_name = dict()

    def __getitem__(self, index):
        x, target = self.dat[index]

        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return x, target

    def __len__(self):
        return len(self.dat)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    from functools import partial

    import torchvision.transforms as trans
    import torchvision.transforms.functional as ftrans

    image_size = 64

    dat = CelebA(
        transform=trans.Compose(
            [
                # trans.RandomHorizontalFlip(),
                partial(ftrans.crop, top=40, left=20, height=218 - 80, width=178 - 40),
                trans.Resize(image_size),
                # trans.CenterCrop(image_size),
                # trans.ToTensor(),
                # trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    )

    x, y = dat[1]
    x.show()

    print("Number of datapoints: {}".format(len(dat)))
