import os
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10, MNIST
from PIL import Image
import medmnist
import wilds
from wilds.datasets.wilds_dataset import WILDSSubset


class CIFAR10Dataset:
    """
    CIFAR-10 Dataset Class.
    """
    def __init__(self, dpath: str, rot_angle: int = 0) -> None:
        self.image_size = 32
        self.NUM_CLS = 10
        self.mean = np.array((0.4914, 0.4822, 0.4465))
        self.std = np.array((0.2023, 0.1994, 0.2010))
        self.num_channels = 3

        self.train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(self.mean, self.std),
        ])
        self.test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(self.mean, self.std),
        ])
        
        if rot_angle != 0:
            self.train_transform.transforms.insert(1, T.RandomVerticalFlip())
            self.train_transform.transforms.append(
                T.Lambda(lambda img: T.functional.rotate(img, rot_angle))
            )
            self.test_transform.transforms.append(
                T.Lambda(lambda img: T.functional.rotate(img, rot_angle))
            )

        self.train_dset = CIFAR10(root=dpath, train=True, download=True, transform=self.train_transform)
        self.test_dset = CIFAR10(root=dpath, train=False, download=True, transform=self.test_transform)
        self.image_bound_l = torch.tensor((-self.mean / self.std).reshape(1, -1, 1, 1)).float()
        self.image_bound_u = torch.tensor(((1 - self.mean) / self.std).reshape(1, -1, 1, 1)).float()


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

        self.train_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(self.mean, self.std),
        ])
        self.test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(self.mean, self.std),
        ])
        self.train_dset = MNIST(root=dpath, train=True, download=True, transform=self.train_transform)
        self.test_dset = MNIST(root=dpath, train=False, download=True, transform=self.test_transform)


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

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(self.mean, self.std)
        ])
        
        if not os.path.exists(dpath):
            os.makedirs(dpath)

        def target_transform(x):
            return x[0]

        self.train_dset = self.data_class(
            root=dpath, split="train", transform=self.transform,
            target_transform=target_transform, download=True
        )
        self.test_dset = self.data_class(
            root=dpath, split="test", transform=self.transform,
            target_transform=target_transform, download=True
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


class CacheDataset:
    """
    Caches the entire dataset in memory.
    """
    def __init__(self, dset):
        self.data = []
        self.targets = getattr(dset, "targets", None)
        for i in range(len(dset)):
            self.data.append(dset[i])

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class TransformDataset:
    """
    Applies a transformation to the dataset.
    """
    def __init__(self, dset, transform):
        self.dset = dset
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.dset[index]
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.dset)


def read_domainnet_data(dataset_path: str, domain_name: str, split: str = "train", labels_to_keep=None):
    """
    Reads DomainNet data.
    """
    data_paths = []
    data_labels = []
    split_file = os.path.join(dataset_path, "splits", f"{domain_name}_{split}.txt")

    with open(split_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data_path, label = line.split(" ")
            label_name = data_path.split("/")[1]
            if labels_to_keep is None or label_name in labels_to_keep:
                data_path = os.path.join(dataset_path, data_path)
                if labels_to_keep is not None:
                    label = labels_to_keep.index(label_name)
                else:
                    label = int(label)
                data_paths.append(data_path)
                data_labels.append(label)

    return data_paths, data_labels


class DomainNet:
    """
    DomainNet Dataset Class.
    """
    def __init__(self, data_paths, data_labels, transforms, domain_name, cache=False):
        self.data_paths = data_paths
        self.data_labels = data_labels
        self.transforms = transforms
        self.domain_name = domain_name
        self.cached_data = []
        
        if cache:
            for idx, _ in enumerate(data_paths):
                self.cached_data.append(self.__read_data__(idx))

    def __read_data__(self, index):
        img = Image.open(self.data_paths[index])
        if img.mode != "RGB":
            img = img.convert("RGB")
        label = self.data_labels[index]
        img = T.ToTensor()(img)
        return img, label

    def __getitem__(self, index):
        if self.cached_data:
            img, label = self.cached_data[index]
        else:
            img, label = self.__read_data__(index)
        img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.data_paths)


class DomainNetDataset:
    """
    DomainNet Dataset Class.
    """
    def __init__(self, dpath: str, domain_name: str) -> None:
        self.image_size = 32
        self.crop_scale = 0.75
        self.image_resize = int(np.ceil(self.image_size / self.crop_scale))

        labels_to_keep = [
            "suitcase", "teapot", "pillow", "streetlight", "table",
            "bathtub", "wine_glass", "vase", "umbrella", "bench"
        ]
        self.num_cls = len(labels_to_keep)
        self.num_channels = 3

        train_transform = T.Compose([
            T.Resize((self.image_resize, self.image_resize), antialias=True),
        ])
        test_transform = T.Compose([
            T.Resize((self.image_size, self.image_size), antialias=True),
        ])
        train_data_paths, train_data_labels = read_domainnet_data(
            dpath, domain_name, split="train", labels_to_keep=labels_to_keep
        )
        test_data_paths, test_data_labels = read_domainnet_data(
            dpath, domain_name, split="test", labels_to_keep=labels_to_keep
        )
        self.train_dset = DomainNet(
            train_data_paths, train_data_labels, train_transform, domain_name
        )
        self.test_dset = DomainNet(
            test_data_paths, test_data_labels, test_transform, domain_name
        )


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
            (dset.metadata_array[:, dset.metadata_fields.index(domain_key)].numpy() == domain) & 
            (dset.split_array == 0)
        )

        self.mean = np.array((0.4914, 0.4822, 0.4465))
        self.std = np.array((0.2023, 0.1994, 0.2010))
        self.num_channels = 3

        train_transform = T.Compose([
            T.RandomResizedCrop(32),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(self.mean, self.std),
        ])
        test_transform = T.Compose([
            T.Resize(32),
            T.ToTensor(),
            T.Normalize(self.mean, self.std),
        ])

        num_samples_domain = len(idx)
        train_samples = int(num_samples_domain * 0.8)
        idx = np.random.permutation(idx)
        train_dset = WILDSSubset(dset, idx[:train_samples], transform=None)
        test_dset = WILDSSubset(dset, idx[train_samples:], transform=None)
        self.train_dset = WildsDset(train_dset, transform=train_transform)
        self.test_dset = CacheDataset(WildsDset(test_dset, transform=test_transform))


def get_dataset(dname: str, dpath: str):
    """
    Returns the appropriate dataset class based on the dataset name.
    """
    dset_mapping = {
        "cifar10": CIFAR10Dataset,
        "cifar10_r0": CIFAR10Dataset,
        "cifar10_r90": CIFAR10R90Dataset,
        "cifar10_r180": CIFAR10R180Dataset,
        "cifar10_r270": CIFAR10R270Dataset,
        "mnist": MNISTDataset,
        "pathmnist": PathMNISTDataset,
        "dermamnist": DermaMNISTDataset,
        "bloodmnist": BloodMNISTDataset,
        "tissuemnist": TissueMNISTDataset,
        "organamnist": OrganAMNISTDataset,
        "organcmnist": OrganCMNISTDataset,
        "organsmnist": OrganSMNISTDataset,
    }

    if dname.startswith("wilds"):
        dname_parts = dname.split("_")
        return WildsDataset(dname_parts[1], dpath, int(dname_parts[2]))
    elif dname.startswith("domainnet"):
        dname_parts = dname.split("_")
        return DomainNetDataset(dpath, dname_parts[1])
    else:
        return dset_mapping[dname](dpath)


def filter_by_class(dataset, classes):
    """
    Filters the dataset by specified classes.
    """
    indices = [i for i, (_, y) in enumerate(dataset) if y in classes]
    return Subset(dataset, indices), indices


def random_samples(dataset, num_samples):
    """
    Returns a random subset of samples from the dataset.
    """
    indices = torch.randperm(len(dataset))[:num_samples]
    return Subset(dataset, indices), indices


def extr_noniid(train_dataset, samples_per_client, classes):
    """
    Extracts non-IID data from the training dataset.
    """
    all_data = Subset(train_dataset, [i for i, (_, y) in enumerate(train_dataset) if y in classes])
    perm = torch.randperm(len(all_data))
    return Subset(all_data, perm[:samples_per_client])


def cifar_extr_noniid(
    train_dataset, test_dataset, num_users, n_class, num_samples, rate_unbalance
):
    """
    Extracts non-IID data for CIFAR-10 dataset.
    """
    num_shards_train = int(50000 / num_samples)
    num_imgs_train = num_samples
    num_classes = 10
    num_imgs_perc_test = 1000
    num_imgs_test_total = 10000

    assert n_class * num_users <= num_shards_train
    assert n_class <= num_classes

    dict_users_train = {i: np.array([]) for i in range(num_users)}
    dict_users_test = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards_train * num_imgs_train)
    labels = np.array(train_dataset.targets)
    idxs_test = np.arange(num_imgs_test_total)
    labels_test = np.array(test_dataset.targets)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]

    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_test = idxs_labels_test[0, :]

    idx_shard = list(range(num_shards_train))

    for i in range(num_users):
        user_labels = np.array([])
        rand_set = set(np.random.choice(idx_shard, n_class, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        unbalance_flag = 0
        for rand in rand_set:
            if unbalance_flag == 0:
                dict_users_train[i] = np.concatenate(
                    (
                        dict_users_train[i],
                        idxs[rand * num_imgs_train: (rand + 1) * num_imgs_train],
                    ),
                    axis=0,
                )
                user_labels = np.concatenate(
                    (
                        user_labels,
                        labels[rand * num_imgs_train: (rand + 1) * num_imgs_train],
                    ),
                    axis=0,
                )
            else:
                dict_users_train[i] = np.concatenate(
                    (
                        dict_users_train[i],
                        idxs[
                            rand * num_imgs_train: int((rand + rate_unbalance) * num_imgs_train)
                        ],
                    ),
                    axis=0,
                )
                user_labels = np.concatenate(
                    (
                        user_labels,
                        labels[
                            rand * num_imgs_train: int((rand + rate_unbalance) * num_imgs_train)
                        ],
                    ),
                    axis=0,
                )
            unbalance_flag = 1
        
        user_labels_set = set(user_labels)
        for label in user_labels_set:
            dict_users_test[i] = np.concatenate(
                (
                    dict_users_test[i],
                    idxs_test[
                        int(label) * num_imgs_perc_test: int(label + 1) * num_imgs_perc_test
                    ],
                ),
                axis=0,
            )

    return dict_users_train, dict_users_test


def balanced_subset(dataset, num_samples):
    """
    Returns a balanced subset of the dataset.
    """
    indices = []
    targets = np.array(dataset.targets)
    classes = set(dataset.targets)
    for c in classes:
        indices += list((targets == c).nonzero()[0][:num_samples])
    indices = np.random.permutation(indices)
    return Subset(dataset, indices), indices


def random_balanced_subset(dataset, num_samples):
    """
    Returns a random balanced subset of the dataset.
    """
    indices = []
    targets = np.array(dataset.targets)
    classes = set(dataset.targets)
    for c in classes:
        indices += list(
            np.random.choice(list((targets == c).nonzero()[0]), num_samples, replace=False)
        )
    return Subset(dataset, indices), indices


def non_iid_unbalanced_dataidx_map(dset_obj, n_parties, beta=0.4):
    """
    Returns a non-IID unbalanced data index map.
    """
    train_dset = dset_obj.train_dset
    n_classes = dset_obj.num_cls

    N = len(train_dset)
    labels = np.array(train_dset.targets)
    min_size = 0
    min_require_size = 10

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_parties)]
        for k in range(n_classes):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    net_dataidx_map = {}
    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
    return net_dataidx_map


def non_iid_balanced(dset_obj, n_client, n_data_per_clnt, alpha=0.4, cls_priors=None, is_train=True):
    """
    Returns a non-IID balanced dataset.
    """
    if is_train:
        y = np.array(dset_obj.train_dset.targets)
    else:
        y = np.array(dset_obj.test_dset.targets)
    
    n_cls = dset_obj.num_cls
    clnt_data_list = (np.ones(n_client) * n_data_per_clnt).astype(int)
    if cls_priors is None:
        cls_priors = np.random.dirichlet(alpha=[alpha] * n_cls, size=n_client)
    
    prior_cumsum = np.cumsum(cls_priors, axis=1)
    idx_list = [np.where(y == i)[0] for i in range(n_cls)]
    cls_amount = np.array([len(idx_list[i]) for i in range(n_cls)])
    clnt_y = [np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(n_client)]
    clnt_idx = [[] for clnt__ in range(n_client)]
    clients = list(np.arange(n_client))

    while np.sum(clnt_data_list) != 0:
        curr_clnt = np.random.choice(clients)
        if clnt_data_list[curr_clnt] <= 0:
            clients.remove(curr_clnt)
            continue
        clnt_data_list[curr_clnt] -= 1
        curr_prior = prior_cumsum[curr_clnt]
        while True:
            cls_label = np.argmax((np.random.uniform() <= curr_prior) & (cls_amount > 0))
            if cls_amount[cls_label] <= 0:
                continue
            cls_amount[cls_label] -= 1
            idx = idx_list[cls_label][cls_amount[cls_label]]
            clnt_y[curr_clnt][clnt_data_list[curr_clnt]] = y[idx]
            clnt_idx[curr_clnt].append(idx)
            break

    clnt_y = np.asarray(clnt_y)
    return clnt_y, clnt_idx, cls_priors
