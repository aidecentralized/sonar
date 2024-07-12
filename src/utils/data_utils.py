import importlib
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Subset



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


def get_dataset(dname: str, dpath: str):
    """
    Returns the appropriate dataset class based on the dataset name.
    """
    dset_mapping = {
        "cifar10": ("data_loaders.cifar", "CIFAR10Dataset"),
        "cifar10_r0": ("data_loaders.cifar", "CIFAR10Dataset"),
        "cifar10_r90": ("data_loaders.cifar", "CIFAR10R90Dataset"),
        "cifar10_r180": ("data_loaders.cifar", "CIFAR10R180Dataset"),
        "cifar10_r270": ("data_loaders.cifar", "CIFAR10R270Dataset"),
        "mnist": ("data_loaders.mnist", "MNISTDataset"),
        "pathmnist": ("data_loaders.medmnist", "PathMNISTDataset"),
        "dermamnist": ("data_loaders.medmnist", "DermaMNISTDataset"),
        "bloodmnist": ("data_loaders.medmnist", "BloodMNISTDataset"),
        "tissuemnist": ("data_loaders.medmnist", "TissueMNISTDataset"),
        "organamnist": ("data_loaders.medmnist", "OrganAMNISTDataset"),
        "organcmnist": ("data_loaders.medmnist", "OrganCMNISTDataset"),
        "organsmnist": ("data_loaders.medmnist", "OrganSMNISTDataset"),
        "domainnet": ("data_loaders.domainnet", "DomainNetDataset"),
        "wilds": ("data_loaders.wilds", "WildsDataset"),
    }

    if dname not in dset_mapping:
        raise ValueError(f"Unknown dataset name: {dname}")

    if dname.startswith("wilds"):
        dname_parts = dname.split("_")
        module_path, class_name = dset_mapping["wilds"]
        module = importlib.import_module(module_path)
        dataset_class = getattr(module, class_name)
        return dataset_class(dname_parts[1], dpath, int(dname_parts[2]))
    elif dname.startswith("domainnet"):
        dname_parts = dname.split("_")
        module_path, class_name = dset_mapping["domainnet"]
        module = importlib.import_module(module_path)
        dataset_class = getattr(module, class_name)
        return dataset_class(dpath, dname_parts[1])
    else:
        module_path, class_name = dset_mapping[dname]
        module = importlib.import_module(module_path)
        dataset_class = getattr(module, class_name)
        return dataset_class(dpath)


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


def  cifar_extr_noniid(
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
