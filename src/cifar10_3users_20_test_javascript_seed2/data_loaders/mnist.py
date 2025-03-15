import torchvision.transforms as T
from torchvision.datasets import MNIST


class MNISTDataset:
    """
    MNIST Dataset Class.
    """

    def __init__(self, dpath: str) -> None:
        self.image_size = 28
        self.num_cls = 10
        self.mean = 0.1307
        self.std = 0.3081
        self.num_channels = 1

        self.train_transform = T.Compose(
            [
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
        self.train_dset = MNIST(
            root=dpath, train=True, download=True, transform=self.train_transform
        )
        self.test_dset = MNIST(
            root=dpath, train=False, download=True, transform=self.test_transform
        )
