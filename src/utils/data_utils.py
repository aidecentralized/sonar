import pdb
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets import MNIST
from torch.utils.data import Subset
import os
from PIL import Image

import medmnist

import wilds
from wilds.datasets.wilds_dataset import WILDSSubset

class CIFAR10_DSET():
    def __init__(self, dpath, rot_angle=0) -> None:
        self.IMAGE_SIZE = 32
        self.NUM_CLS = 10
        self.mean = np.array((0.4914, 0.4822, 0.4465))
        self.std = np.array((0.2023, 0.1994, 0.2010))
        self.num_channels = 3
        self.gen_transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.Normalize(
                    self.mean, 
                    self.std
                ),
            ]
        )
        train_transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip() if rot_angle % 180 == 0 else T.RandomVerticalFlip(),
                T.ToTensor(),
                T.Normalize(
                    self.mean, 
                    self.std
                ),
            ]
        )
        test_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    self.mean, 
                    self.std
                ),
            ]
        )
        if rot_angle != 0:
            tr_transform, te_transform = train_transform, test_transform
            train_transform = lambda img: T.functional.rotate(tr_transform(img), angle=rot_angle)
            test_transform = lambda img: T.functional.rotate(te_transform(img), angle=rot_angle)
        
        self.train_dset = CIFAR10(
            root=dpath, train=True, download=True, transform=train_transform
        )
        self.test_dset = CIFAR10(
            root=dpath, train=False, download=True, transform=test_transform
        )
        self.IMAGE_BOUND_L = torch.tensor((-self.mean / self.std).reshape(1, -1, 1, 1)).float()
        self.IMAGE_BOUND_U = torch.tensor(((1 - self.mean) / self.std).reshape(1, -1, 1, 1)).float()

class CIFAR10_R90_DSET(CIFAR10_DSET):
    def __init__(self, dpath) -> None:
        super().__init__(dpath, rot_angle=90)
        
class CIFAR10_R180_DSET(CIFAR10_DSET):
    def __init__(self, dpath) -> None:
        super().__init__(dpath, rot_angle=180)
        
class CIFAR10_R270_DSET(CIFAR10_DSET):
    def __init__(self, dpath) -> None:
        super().__init__(dpath, rot_angle=270)
        
class MNIST_DSET():
    def __init__(self, dpath) -> None:
        self.IMAGE_SIZE = 28
        self.NUM_CLS = 10
        self.mean = 0.1307
        self.std = 0.3081
        self.num_channels = 1
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
                T.ToTensor(),
                T.Normalize(
                    self.mean, 
                    self.std
                ),
            ]
        )
        test_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    self.mean, 
                    self.std
                ),
            ]
        )
        self.train_dset = MNIST(
            root=dpath, train=True, download=True, transform=train_transform
        )
        self.test_dset = MNIST(
            root=dpath, train=False, download=True, transform=test_transform
        )

class MEDMNIST_DSET():
    def __init__(self, dpath, data_flag) -> None:
        self.mean = np.array([0.5])
        self.std = np.array([0.5])
        info = medmnist.INFO[data_flag]
        self.num_channels = info['n_channels']
        self.data_class = getattr(medmnist, info['python_class'])
        transform = T.Compose([T.ToTensor(), T.Normalize(mean=[.5], std=[.5])])
        if not os.path.exists(dpath):
            os.makedirs(dpath)
            
        target_transform = lambda x: x[0]
        self.train_dset = self.data_class(root=dpath, split='train', transform=transform, target_transform=target_transform, download=True)
        self.test_dset = self.data_class(root=dpath, split='test', transform=transform, target_transform=target_transform, download=True)

class PathMNIST_DSET(MEDMNIST_DSET):
    def __init__(self, dpath) -> None:
        super().__init__(dpath, "pathmnist")

        self.IMAGE_SIZE = 28
        self.NUM_CLS = 9
  
class DermaMNIST_DSET(MEDMNIST_DSET):
    def __init__(self, dpath) -> None:
        super().__init__(dpath, "dermamnist")

        self.IMAGE_SIZE = 28
        self.NUM_CLS = 7  
        
class BloodMNIST_DSET(MEDMNIST_DSET):
    def __init__(self, dpath) -> None:
        super().__init__(dpath, "bloodmnist")

        self.IMAGE_SIZE = 28
        self.NUM_CLS = 8  
        
class TissueMNIST_DSET(MEDMNIST_DSET):
    def __init__(self, dpath) -> None:
        super().__init__(dpath, "tissuemnist")

        self.IMAGE_SIZE = 28
        self.NUM_CLS = 8 

class OrganAMNIST_DSET(MEDMNIST_DSET):
    def __init__(self, dpath) -> None:
        super().__init__(dpath, "organamnist")

        self.IMAGE_SIZE = 28
        self.NUM_CLS = 11 
        
class OrganCMNIST_DSET(MEDMNIST_DSET):
    def __init__(self, dpath) -> None:
        super().__init__(dpath, "organcmnist")

        self.IMAGE_SIZE = 28
        self.NUM_CLS = 11
        
class OrganSMNIST_DSET(MEDMNIST_DSET):
    def __init__(self, dpath) -> None:
        super().__init__(dpath, "organsmnist")

        self.IMAGE_SIZE = 28
        self.NUM_CLS = 11
  
class CacheDataset():
    def __init__(self, dset):
        
        if hasattr(dset, 'targets'):
            self.targets = dset.targets
        
        self.data = []
        for i in range(len(dset)):
            self.data.append(dset[i])
        
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
  
class TransformDataset():
    def __init__(self, dset, transform):
        self.dset = dset
        self.transform = transform
        
    def __getitem__(self, index):
        img, label = self.dset[index]
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.dset) 
    
# https://github.com/FengHZ/KD3A/blob/master/datasets/DomainNet.py
def read_domainnet_data(dataset_path, domain_name, split="train", labels_to_keep=None):
    data_paths = []
    data_labels = []
    split_file = os.path.join(dataset_path, "splits", "{}_{}.txt".format(domain_name, split))
    with open(split_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data_path, label = line.split(' ')
            label_name = data_path.split('/')[1]
            if labels_to_keep is None or label_name in labels_to_keep:
                data_path = os.path.join(dataset_path, data_path)
                if labels_to_keep is not None:
                    label = labels_to_keep.index(label_name)
                else:
                    label = int(label)
                data_paths.append(data_path)
                data_labels.append(label)
    return data_paths, data_labels    
      
class DomainNet():
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
        if not img.mode == "RGB":
            img = img.convert("RGB")
        label = self.data_labels[index]
        img = T.ToTensor()(img)
        
        return img, label

    def __getitem__(self, index):
        if len(self.cached_data) > 0:
            img, label = self.cached_data[index]
        else:
            img, label = self.__read_data__(index)
        img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.data_paths)
        
class DomainNet_DSET():
    def __init__(self, dpath, domain_name):
        # TODO Modify ResNet to support 64 x 64 images
        self.IMAGE_SIZE = 32
        self.CROP_SCALE = 0.75
        self.IMAZE_RESIZE = int(np.ceil(self.IMAGE_SIZE * 1 / self.CROP_SCALE))

        labels_to_keep = ["suitcase", "teapot", "pillow", "streetlight", "table", "bathtub", "wine_glass", "vase", "umbrella", "bench"]
        self.NUM_CLS = len(labels_to_keep)
        self.num_channels = 3

        train_transform =  T.Compose([
            T.Resize((self.IMAZE_RESIZE, self.IMAZE_RESIZE), antialias=True),
            # T.ToTensor()
        ])
        test_transform =  T.Compose([
            T.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE), antialias=True),
            # T.ToTensor()
        ])
        train_data_paths, train_data_labels = read_domainnet_data(dpath, domain_name, split="train", labels_to_keep=labels_to_keep)
        test_data_paths, test_data_labels = read_domainnet_data(dpath, domain_name, split="test", labels_to_keep=labels_to_keep)
        self.train_dset = DomainNet(train_data_paths, train_data_labels, train_transform, domain_name)
        self.test_dset = DomainNet(test_data_paths, test_data_labels, test_transform, domain_name)
          
WILDS_DOMAINS_DICT ={
    "iwildcam": "location",
    "camelyon17": "hospital",
    "rxrx1": "experiment",
    "fmow": "region",
}
  
class WildsDset():
    def __init__(self, dset, transform=None):
        self.dset = dset
        self.transform = transform
        self.targets = [t.item() for t in list(dset.y_array)]
    
    def __getitem__(self, index):
        img, label, meta_data = self.dset[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label.item()
    
    def __len__(self):
        return len(self.dset)
  
class Wilds_DSET():
    def __init__(self, dset_name, dpath, domain):
        dset = wilds.get_dataset(dset_name, download=False, root_dir=dpath)
        self.NUM_CLS = len(list(np.unique(dset.y_array)))

        # print("Dataset: ", len(dset))
        # print("Number of classes: ",self.NUM_CLS)
        # # print("Split arrays", np.unique(dset.split_array))
        # # print("Meta", np.unique(dset.metadata_array[:, dset.metadata_fields.index(WILDS_DOMAINS_DICT[dset_name])].numpy()))
        
        # print(dset.metadata_fields)
        # print(np.unique(dset.metadata_array[:, dset.metadata_fields.index("region")].numpy()))
        # print(np.unique(dset.metadata_array[:, dset.metadata_fields.index("year")].numpy()))

        # for i in range(51):
        #     idx, = np.where(np.logical_and(dset.metadata_array[:, dset.metadata_fields.index(WILDS_DOMAINS_DICT[dset_name])].numpy()==i,
        #                        dset.split_array==0))
        #     print("Domain: ", i, "Train samples: ", len(idx))
        
        # Most wilds dset only have OOD data in the test set so we use the train set for both train and test
        idx, = np.where(np.logical_and(dset.metadata_array[:, dset.metadata_fields.index(WILDS_DOMAINS_DICT[dset_name])].numpy()==domain,
                                dset.split_array==0))
        
        # print("Dataset filter: ", len(idx))                
        self.mean = np.array((0.4914, 0.4822, 0.4465))
        self.std = np.array((0.2023, 0.1994, 0.2010))
        self.num_channels = 3

        train_transform = T.Compose(
            [
                T.RandomResizedCrop(32),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(
                    self.mean, 
                    self.std
                ),
            ]
        )
        test_transform = T.Compose(
            [
                T.Resize(32),
                T.ToTensor(),
                T.Normalize(
                    self.mean, 
                    self.std
                ),
            ]
        )

        num_samples_domain = len(idx)
        TRAIN_RATIO = 0.8
        train_samples = int(num_samples_domain * TRAIN_RATIO)
        idx = np.random.permutation(idx)
        train_dset = WILDSSubset(dset, idx[:train_samples], transform=None)
        test_dset = WILDSSubset(dset, idx[train_samples:], transform=None)
        self.train_dset = WildsDset(train_dset, transform=train_transform)
        self.test_dset = CacheDataset(WildsDset(test_dset, transform=test_transform))
         
def get_dataset(dname, dpath):
    dset_mapping = {"cifar10": CIFAR10_DSET,
                    "cifar10_r0": CIFAR10_DSET,
                    "cifar10_r90": CIFAR10_R90_DSET,
                    "cifar10_r180": CIFAR10_R180_DSET,
                    "cifar10_r270": CIFAR10_R270_DSET,
                    "mnist": MNIST_DSET,
                    # "cifar100": CIFAR100_DSET,
                    "pathmnist": PathMNIST_DSET,
                    "dermamnist":DermaMNIST_DSET,
                    "bloodmnist":BloodMNIST_DSET,
                    "tissuemnist":BloodMNIST_DSET,
                    "organamnist":OrganAMNIST_DSET,
                    "organcmnist":OrganCMNIST_DSET,
                    "organsmnist":OrganSMNIST_DSET,
                    }
    
    if dname.startswith("wilds"):
        dname = dname.split("_")
        return Wilds_DSET(dname[1], dpath, int(dname[2]))    
    elif dname.startswith("domainnet"):
        dname = dname.split("_")
        return DomainNet_DSET(dpath, dname[1])
    else:
        return dset_mapping[dname](dpath)

"""def get_noniid_dataset(dname, dpath, num_users, n_class, nsamples, rate_unbalance):
    obj = get_dataset(dname, dpath)
    # Chose euqal splits for every user
    if dname == "cifar10":
        obj.user_groups_train, obj.user_groups_test = cifar_extr_noniid(obj.train_dset, obj.test_dset,
                                                                        num_users, n_class, nsamples,
                                                                        rate_unbalance)
    return obj"""

def filter_by_class(dataset, classes):
    indices = [i for i,(x, y) in enumerate(dataset) if y in classes]
    return Subset(dataset, indices), indices

def random_samples(dataset, num_samples):
    indices = torch.randperm(len(dataset))[:num_samples]
    return Subset(dataset, indices), indices 

def extr_noniid(train_dataset, samples_per_client, classes):
    all_data=Subset(train_dataset,[i for i,(x, y) in enumerate(train_dataset) if y in classes])
    perm=torch.randperm(len(all_data))
    return Subset(all_data,perm[:samples_per_client])

def cifar_extr_noniid(train_dataset, test_dataset, num_users, n_class, num_samples, rate_unbalance):
    num_shards_train, num_imgs_train = int(50000/num_samples), num_samples
    num_classes = 10
    num_imgs_perc_test, num_imgs_test_total = 1000, 10000
    assert(n_class * num_users <= num_shards_train)
    assert(n_class <= num_classes)
    idx_class = [i for i in range(num_classes)]
    idx_shard = [i for i in range(num_shards_train)]
    dict_users_train = {i: np.array([]) for i in range(num_users)}
    dict_users_test = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards_train*num_imgs_train)
    # labels = dataset.train_labels.numpy()
    labels = np.array(train_dataset.targets)
    idxs_test = np.arange(num_imgs_test_total)
    labels_test = np.array(test_dataset.targets)
    #labels_test_raw = np.array(test_dataset.targets)

    # stores the image idxs with their corresponding labels
    # array([[    0,     1,     2, ..., 49997, 49998, 49999],
    #        [    6,     9,     9, ...,     9,     1,     1]])
    idxs_labels = np.vstack((idxs, labels))
    # sorts the whole thing based on labels
    # array([[29513, 16836, 32316, ..., 36910, 21518, 25648],
    #       [    0,     0,     0, ...,     9,     9,     9]])
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]

    # Same things as above except that it is test set now
    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_test = idxs_labels_test[0, :]
    #print(idxs_labels_test[1, :])


    # divide and assign
    for i in range(num_users):
        user_labels = np.array([])
        rand_set = set(np.random.choice(idx_shard, n_class, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        unbalance_flag = 0
        for rand in rand_set:
            if unbalance_flag == 0:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], idxs[rand*num_imgs_train:(rand+1)*num_imgs_train]), axis=0)
                user_labels = np.concatenate((user_labels, labels[rand*num_imgs_train:(rand+1)*num_imgs_train]), axis=0)
            else:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], idxs[rand*num_imgs_train:int((rand+rate_unbalance)*num_imgs_train)]), axis=0)
                user_labels = np.concatenate((user_labels, labels[rand*num_imgs_train:int((rand+rate_unbalance)*num_imgs_train)]), axis=0)
            unbalance_flag = 1
        user_labels_set = set(user_labels)
        #print(user_labels_set)
        #print(user_labels)
        for label in user_labels_set:
            dict_users_test[i] = np.concatenate((dict_users_test[i], idxs_test[int(label)*num_imgs_perc_test:int(label+1)*num_imgs_perc_test]), axis=0)
        #print(set(labels_test_raw[dict_users_test[i].astype(int)]))

    return dict_users_train, dict_users_test

def balanced_subset(dataset, num_samples):
    indices = []
    targets = np.array(dataset.targets)
    classes = set(dataset.targets)
    for c in classes:
        indices += list((targets == c).nonzero()[0][:num_samples])
        
    # Avoid samples from the same class being consecutive
    indices = np.random.permutation(indices)
    return Subset(dataset, indices), indices

def random_balanced_subset(dataset, num_samples):
    indices = []
    targets = np.array(dataset.targets)
    classes = set(dataset.targets)
    for c in classes:
        indices += list(np.random.choice(list((targets == c).nonzero()[0]), num_samples, replace=False))
    return Subset(dataset, indices), indices

def non_iid_unbalanced_dataidx_map(dset_obj, n_parties, beta=0.4):
    train_dset = dset_obj.train_dset
    n_classes = dset_obj.NUM_CLS
    
    N = len(train_dset)
    labels = np.array(train_dset.targets)
    min_size = 0 # Tracks the minimum number of samples in a party
    min_require_size = 10
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_parties)]
        for k in range(n_classes):
            # Get indexes of class k
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            
            # Sample proportions from a dirichlet distribution
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            
            # Keep only proportions that lead to a samller number of samples than 
            proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            
            # Get range of split according to proportions
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            # Divide class k indexes according to proportions
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    # Convert list to map
    net_dataidx_map = {}        
    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
    return net_dataidx_map

def non_iid_balanced(dset_obj, n_client, n_data_per_clnt, alpha=0.4, cls_priors=None, is_train=True):
    if is_train:
        # y, x = np.array(dset_obj.train_dset.targets), np.array(dset_obj.train_dset.data)
        y = np.array(dset_obj.train_dset.targets)
    else:
        #y, x = np.array(dset_obj.test_dset.targets), np.array(dset_obj.test_dset.data)
        y = np.array(dset_obj.test_dset.targets)
    n_cls = dset_obj.NUM_CLS
    height = width = dset_obj.IMAGE_SIZE
    channels = dset_obj.num_channels

    clnt_data_list = (np.ones(n_client) * n_data_per_clnt).astype(int) # Number of data per client
    if cls_priors is None:
        cls_priors = np.random.dirichlet(alpha=[alpha]*n_cls,size=n_client) 
    prior_cumsum = np.cumsum(cls_priors, axis=1)    
    idx_list = [np.where(y==i)[0] for i in range(n_cls)]
    cls_amount = np.array([len(idx_list[i]) for i in range(n_cls)])

    #clnt_x = [np.zeros((clnt_data_list[clnt__], height, width, channels)).astype(np.float32) for clnt__ in range(n_client) ]
    clnt_y = [np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(n_client) ]
    clnt_idx = [[] for clnt__ in range(n_client)]
    clients = list(np.arange(n_client))
    while(np.sum(clnt_data_list)!=0):
        curr_clnt = np.random.choice(clients)
        #curr_clnt = np.random.randint(n_client)
        # If current node is full resample a client
        # print('Remaining Data: %d' %np.sum(clnt_data_list))
        if clnt_data_list[curr_clnt] <= 0:
            clients.remove(curr_clnt)
            continue
        clnt_data_list[curr_clnt] -= 1
        curr_prior = prior_cumsum[curr_clnt]
        while True:
            cls_label = np.argmax((np.random.uniform() <= curr_prior) & (cls_amount > 0))
            # Redraw class label if trn_y is out of that class
            if cls_amount[cls_label] <= 0:
                continue
            cls_amount[cls_label] -= 1
            idx = idx_list[cls_label][cls_amount[cls_label]]
            #clnt_x[curr_clnt][clnt_data_list[curr_clnt]] = x[idx]
            clnt_y[curr_clnt][clnt_data_list[curr_clnt]] = y[idx]
            clnt_idx[curr_clnt].append(idx)
            break
    #clnt_x = np.asarray(clnt_x)
    clnt_y = np.asarray(clnt_y)
    
    return clnt_y, clnt_idx, cls_priors

