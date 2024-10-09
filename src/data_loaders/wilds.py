import numpy as np
import wilds
import torchvision.transforms as T
from wilds.datasets.wilds_dataset import WILDSSubset
from utils.data_utils import CacheDataset


WILDS_DOMAINS_DICT = {
    "iwildcam": "location",
    "camelyon17": "hospital",
    "rxrx1": "experiment",
    "fmow": "region",
}


class WildsDset:
    """
    WILDS Dataset Class.
    """

    def __init__(self, dset, transform=None):
        self.dset = dset
        self.transform = transform
        self.targets = [t.item() for t in list(dset.y_array)]

    def __getitem__(self, index):
        img, label, _ = self.dset[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label.item()

    def __len__(self):
        return len(self.dset)


class WildsDataset:
    """
    WILDS Dataset Class.
    """

    def __init__(self, dset_name: str, dpath: str, domain: int) -> None:
        dset = wilds.get_dataset(dset_name, download=False, root_dir=dpath)
        self.num_cls = len(list(np.unique(dset.y_array)))

        domain_key = WILDS_DOMAINS_DICT[dset_name]
        (idx,) = np.where(
            (
                dset.metadata_array[:, dset.metadata_fields.index(domain_key)].numpy()
                == domain
            )
            & (dset.split_array == 0)
        )

        self.mean = np.array((0.4914, 0.4822, 0.4465))
        self.std = np.array((0.2023, 0.1994, 0.2010))
        self.num_channels = 3

        train_transform = T.Compose(
            [
                T.RandomResizedCrop(32),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        test_transform = T.Compose(
            [
                T.Resize(32),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )

        num_samples_domain = len(idx)
        train_samples = int(num_samples_domain * 0.8)
        idx = np.random.permutation(idx)
        train_dset = WILDSSubset(dset, idx[:train_samples], transform=None)
        test_dset = WILDSSubset(dset, idx[train_samples:], transform=None)
        self.train_dset = WildsDset(train_dset, transform=train_transform)
        self.test_dset = CacheDataset(WildsDset(test_dset, transform=test_transform))
