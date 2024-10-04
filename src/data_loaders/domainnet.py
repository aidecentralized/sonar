import os
import numpy as np
from PIL import Image
import torchvision.transforms as T


def read_domainnet_data(
    dataset_path: str, domain_name: str, split: str = "train", labels_to_keep=None
):
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
            "suitcase",
            "teapot",
            "pillow",
            "streetlight",
            "table",
            "bathtub",
            "wine_glass",
            "vase",
            "umbrella",
            "bench",
        ]
        self.num_cls = len(labels_to_keep)
        self.num_channels = 3

        train_transform = T.Compose(
            [
                T.Resize((self.image_resize, self.image_resize), antialias=True),
            ]
        )
        test_transform = T.Compose(
            [
                T.Resize((self.image_size, self.image_size), antialias=True),
            ]
        )
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
