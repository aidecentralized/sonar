import os
import medmnist
import numpy as np
import torchvision.transforms as T


class MEDMNISTDataset:
    """
    MEDMNIST Dataset Class.
    """

    def __init__(self, dpath: str, data_flag: str) -> None:
        self.mean = np.array([0.5])
        self.std = np.array([0.5])
        info = medmnist.INFO[data_flag]
        self.num_channels = info["n_channels"]
        self.data_class = getattr(medmnist, info["python_class"])

        self.transform = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])

        if not os.path.exists(dpath):
            os.makedirs(dpath)

        def target_transform(x):
            return x[0]

        self.train_dset = self.data_class(
            root=dpath,
            split="train",
            transform=self.transform,
            target_transform=target_transform,
            download=True,
        )
        self.test_dset = self.data_class(
            root=dpath,
            split="test",
            transform=self.transform,
            target_transform=target_transform,
            download=True,
        )


class PathMNISTDataset(MEDMNISTDataset):
    """
    PathMNIST Dataset Class.
    """

    def __init__(self, dpath: str) -> None:
        super().__init__(dpath, "pathmnist")
        self.image_size = 28
        self.num_cls = 9


class DermaMNISTDataset(MEDMNISTDataset):
    """
    DermaMNIST Dataset Class.
    """

    def __init__(self, dpath: str) -> None:
        super().__init__(dpath, "dermamnist")
        self.image_size = 28
        self.num_cls = 7


class BloodMNISTDataset(MEDMNISTDataset):
    """
    BloodMNIST Dataset Class.
    """

    def __init__(self, dpath: str) -> None:
        super().__init__(dpath, "bloodmnist")
        self.image_size = 28
        self.num_cls = 8


class TissueMNISTDataset(MEDMNISTDataset):
    """
    TissueMNIST Dataset Class.
    """

    def __init__(self, dpath: str) -> None:
        super().__init__(dpath, "tissuemnist")
        self.image_size = 28
        self.num_cls = 8


class OrganAMNISTDataset(MEDMNISTDataset):
    """
    OrganAMNIST Dataset Class.
    """

    def __init__(self, dpath: str) -> None:
        super().__init__(dpath, "organamnist")
        self.image_size = 28
        self.num_cls = 11


class OrganCMNISTDataset(MEDMNISTDataset):
    """
    OrganCMNIST Dataset Class.
    """

    def __init__(self, dpath: str) -> None:
        super().__init__(dpath, "organcmnist")
        self.image_size = 28
        self.num_cls = 11


class OrganSMNISTDataset(MEDMNISTDataset):
    """
    OrganSMNIST Dataset Class.
    """

    def __init__(self, dpath: str) -> None:
        super().__init__(dpath, "organsmnist")
        self.image_size = 28
        self.num_cls = 11
