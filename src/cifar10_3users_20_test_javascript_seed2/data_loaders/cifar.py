import numpy as np
import torch
import torchvision.transforms as T
from torchvision.datasets import CIFAR10


class CIFAR10Dataset:
    """
    CIFAR-10 Dataset Class.
    """

    def __init__(self, dpath: str, rot_angle: int = 0) -> None:
        self.image_size = 32
        self.num_cls = 10
        self.mean = np.array((0.4914, 0.4822, 0.4465))
        self.std = np.array((0.2023, 0.1994, 0.2010))
        self.num_channels = 3

        self.train_transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        self.test_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )

        if rot_angle != 0:
            self.train_transform.transforms.insert(1, T.RandomVerticalFlip())
            self.train_transform.transforms.append(
                T.Lambda(lambda img: T.functional.rotate(img, rot_angle))
            )
            self.test_transform.transforms.append(
                T.Lambda(lambda img: T.functional.rotate(img, rot_angle))
            )

        self.train_dset = CIFAR10(
            root=dpath, train=True, download=True, transform=self.train_transform
        )
        self.test_dset = CIFAR10(
            root=dpath, train=False, download=True, transform=self.test_transform
        )
        self.image_bound_l = torch.tensor(
            (-self.mean / self.std).reshape(1, -1, 1, 1)
        ).float()
        self.image_bound_u = torch.tensor(
            ((1 - self.mean) / self.std).reshape(1, -1, 1, 1)
        ).float()


class CIFAR10R90Dataset(CIFAR10Dataset):
    """
    CIFAR-10 Dataset Class with 90 degrees rotation.
    """

    def __init__(self, dpath: str) -> None:
        super().__init__(dpath, rot_angle=90)


class CIFAR10R180Dataset(CIFAR10Dataset):
    """
    CIFAR-10 Dataset Class with 180 degrees rotation.
    """

    def __init__(self, dpath: str) -> None:
        super().__init__(dpath, rot_angle=180)


class CIFAR10R270Dataset(CIFAR10Dataset):
    """
    CIFAR-10 Dataset Class with 270 degrees rotation.
    """

    def __init__(self, dpath: str) -> None:
        super().__init__(dpath, rot_angle=270)
