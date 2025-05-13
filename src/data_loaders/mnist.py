import pdb
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import Subset
import os
from PIL import Image
import scipy.io

class MNISTDataset():
    def __init__(self, dpath="") -> None:
        self.image_size = 32 # original 28
        self.num_cls = 10
        self.mean = 0.1307
        self.std = 0.3081
        self.num_channels = 3 # Originally 1
        self.gen_transform = T.Compose(
            [
                T.Normalize(
                    self.mean, 
                    self.std
                ),
            ]
        )
        train_transform = T.Compose(
            [
                T.Resize(self.image_size),
                T.ToTensor(),
                T.Normalize(
                    self.mean, 
                    self.std
                ),
                self.__copy_channels__,
            ]
        )
        test_transform = T.Compose(
            [
                T.Resize(self.image_size),
                T.ToTensor(),
                T.Normalize(
                    self.mean, 
                    self.std
                ),
                self.__copy_channels__,
            ]
        )
        self.train_dset = MNIST(
            root=dpath, train=True, download=True, transform=train_transform
        )
        self.test_dset = MNIST(
            root=dpath, train=False, download=True, transform=test_transform
        )

    def __copy_channels__(self,x):
            return x.repeat(3, 1, 1)