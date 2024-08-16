from abc import ABC, abstractmethod
import torch
import numpy
from torch.utils.data import DataLoader, Subset

from collections import OrderedDict
from typing import Any, Dict, List, Optional
from torch import Tensor
import copy
import random
import numpy as np

from utils.plot_utils import PlotUtils
from utils.comm_utils import CommUtils
from utils.data_utils import (
    random_samples,
    filter_by_class,
    get_dataset,
    non_iid_balanced,
    balanced_subset,
    CacheDataset,
    TransformDataset,
)
from utils.log_utils import LogUtils
from utils.model_utils import ModelUtils
from utils.community_utils import (
    get_random_communities,
    get_dset_balanced_communities,
    get_dset_communities,
)
import torchvision.transforms as T
import os

from yolo import YOLOLoss

class BaseNode(ABC):
    def __init__(self, config) -> None:
        self.comm_utils = CommUtils()
        self.node_id = self.comm_utils.rank

        if self.node_id == 0:
            self.log_dir = config['log_path']
            config['log_path'] = f'{self.log_dir}/server'
            try:
                os.mkdir(config['log_path'])
            except FileExistsError:
                pass
            config['load_existing'] = False
            self.log_utils = LogUtils(config)
            self.log_utils.log_console("Config: {}".format(config))
            self.plot_utils = PlotUtils(config)

        # Support user specific dataset
        if isinstance(config["dset"], dict):
            if self.node_id != 0:
                config["dset"].pop("0")
            self.dset = config["dset"][str(self.node_id)]
            config["dpath"] = config["dpath"][self.dset]
        else:
            self.dset = config["dset"]

        self.setup_cuda(config)
        self.model_utils = ModelUtils()

        self.dset_obj = get_dataset(self.dset, dpath=config["dpath"])
        self.set_constants()

    def set_constants(self):
        self.best_acc = 0.0

    def setup_cuda(self, config):
        # Need a mapping from rank to device id
        device_ids_map = config["device_ids"]
        node_name = "node_{}".format(self.node_id)
        self.device_ids = device_ids_map[node_name]
        gpu_id = self.device_ids[0]

        if torch.cuda.is_available():
            self.device = torch.device("cuda:{}".format(gpu_id))
            print("Using GPU: cuda:{}".format(gpu_id))
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

    def set_model_parameters(self, config):
        # Model related parameters
        optim_name = config.get("optimizer", "adam")
        if optim_name == "adam":
            optim = torch.optim.Adam
        elif optim_name == "sgd":
            optim = torch.optim.SGD
        else:
            raise ValueError("Unknown optimizer: {}.".format(optim_name))
        num_classes = self.dset_obj.num_cls
        num_channels = self.dset_obj.num_channels
        self.model = self.model_utils.get_model(
            config["model"],
            self.dset,
            self.device,
            self.device_ids,
            num_classes=num_classes,
            num_channels=num_channels,
            pretrained=config.get("pretrained", False),
        )
        self.optim = optim(
            self.model.parameters(),
            lr=config["model_lr"],
            weight_decay=config.get("weight_decay", 0),
        )
        if config.get('dset') == "pascal":
            self.loss_fn = YOLOLoss()
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()

    def set_shared_exp_parameters(self, config):

        if self.node_id != 0:
            community_type, number_of_communities = config.get(
                "community_type", None
            ), config.get("num_communities", 1)
            num_dset = (
                1
                if not isinstance(config["dset"], dict)
                else len(set(config["dset"].values()))
            )
            if community_type is not None and community_type == "dataset":
                self.communities = get_dset_communities(config["num_users"], num_dset)
            elif community_type is None or number_of_communities == 1:
                all_users = list(range(1, config["num_users"] + 1))
                self.communities = {user: all_users for user in all_users}
            elif community_type == "random":
                self.communities = get_random_communities(
                    config["num_users"], number_of_communities
                )
            elif community_type == "balanced":
                num_dset = (
                    1
                    if not isinstance(config["dset"], dict)
                    else len(set(config["dset"].values()))
                )
                # Assume users ordered by dataset and same number of users
                # per dataset
                self.communities = get_dset_balanced_communities(
                    config["num_users"], number_of_communities, num_dset
                )
            else:
                raise ValueError("Unknown community type: {}.".format(community_type))
        if self.node_id == 0:
            self.log_utils.log_console("Communities: {}".format(self.communities))

    @abstractmethod
    def run_protocol(self):
        raise NotImplementedError


class BaseClient(BaseNode):
    """
    Abstract class for all algorithms
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.server_node = 0
        self.set_parameters(config)

    def set_parameters(self, config):
        """
        Set the parameters for the user
        """

        # Set same seed for all users for data distribution and shared
        # parameters so that all users have the same data splits and shared
        # params
        seed = config["seed"]
        torch.manual_seed(seed)
        random.seed(seed)
        numpy.random.seed(seed)

        self.set_shared_exp_parameters(config)

        # Set different seeds for different domain (dataset)
        if isinstance(config["dset"], dict):
            seed += sorted(config["dset"].values()).index(
                config["dset"][str(self.node_id)]
            )
            torch.manual_seed(seed)
            random.seed(seed)
            numpy.random.seed(seed)

        self.set_data_parameters(config)
        # Number of random operation not the same across users with different datasets
        # Random state is not expected to be the same across users after this
        # point

        # Use different seeds for the rest of the experiment
        # Add rank so that every node has a different seed
        seed = config["seed"] + self.node_id
        torch.manual_seed(seed)
        random.seed(seed)
        numpy.random.seed(seed)

        self.set_model_parameters(config)

        # If leader_mode on all node need to choose same random node at each
        # round
        if config.get("leader_mode", False):
            seed = config["seed"]
            torch.manual_seed(seed)
            random.seed(seed)
            numpy.random.seed(seed)

    def set_data_parameters(self, config):

        # Train set and test set from original dataset
        train_dset = self.dset_obj.train_dset
        test_dset = self.dset_obj.test_dset

        print("num train", len(train_dset))
        print("num test", len(test_dset))

        if config.get("test_samples_per_class", None) is not None:
            test_dset, _ = balanced_subset(test_dset, config["test_samples_per_class"])

        samples_per_user = config["samples_per_user"]
        batch_size = config["batch_size"]
        print(f"samples per user: {samples_per_user}, batch size: {batch_size}")

        # Support user specific dataset
        if isinstance(config["dset"], dict):

            def is_same_dest(dset):
                # Consider all variations of cifar10 as the same dataset
                # To avoid having exactly same original dataset (without
                # considering transformation) on multiple users
                if self.dset == "cifar10" or self.dset.startswith("cifar10_"):
                    return dset == "cifar10" or dset.startswith("cifar10_")
                else:
                    return dset == self.dset

            users_with_same_dset = sorted(
                [int(k) for k, v in config["dset"].items() if is_same_dest(v)]
            )
        else:
            users_with_same_dset = list(range(1, config["num_users"] + 1))
        user_idx = users_with_same_dset.index(self.node_id)

        cls_prior = None
        # If iid, each user has random samples from the whole dataset (no
        # overlap between users)
        if config["train_label_distribution"] == "iid":
            indices = numpy.random.permutation(len(train_dset))
            train_indices = indices[
                user_idx * samples_per_user : (user_idx + 1) * samples_per_user
            ]
            train_dset = Subset(train_dset, train_indices)
            classes = list(set([train_dset[i][1] for i in range(len(train_dset))]))
        # If non_iid, each user get random samples from its support classes
        # (mulitple users might have same images)
        elif config["train_label_distribution"] == "support":
            classes = config["support"][str(self.node_id)]
            support_classes_dataset, indices = filter_by_class(train_dset, classes)
            train_dset, sel_indices = random_samples(
                support_classes_dataset, samples_per_user
            )
            train_indices = [indices[i] for i in sel_indices]
        elif config["train_label_distribution"].endswith("non_iid"):
            alpha = config.get("alpha_data", 0.4)
            if config["train_label_distribution"] == "inter_domain_non_iid":
                # Hack to get the same class prior for all users with the same dataset
                # While keeping the same random state for all users
                if isinstance(config["dset"], dict) and isinstance(
                    config["dset"], dict
                ):
                    cls_priors = []
                    dsets = list(config["dset"].values())
                    for _ in dsets:
                        n_cls = self.dset_obj.num_cls
                        cls_priors.append(
                            np.random.dirichlet(
                                alpha=[alpha] * n_cls, size=len(users_with_same_dset)
                            )
                        )
                    cls_prior = cls_priors[dsets.index(self.dset)]
            train_y, train_idx_split, cls_prior = non_iid_balanced(
                self.dset_obj,
                len(users_with_same_dset),
                samples_per_user,
                alpha,
                cls_priors=cls_prior,
                is_train=True,
            )
            train_indices = train_idx_split[self.node_id - 1]
            train_dset = Subset(train_dset, train_indices)
            classes = numpy.unique(train_y[user_idx]).tolist()
            # One plot per dataset
            # if user_idx == 0:
            #     print("using non_iid_balanced", alpha)
            #     self.plot_utils.plot_training_distribution(train_y, self.dset, users_with_same_dset)
        elif config["train_label_distribution"] == "shard":
            raise NotImplementedError
            # classes_per_user = config["shards"]["classes_per_user"]
            # samples_per_shard = samples_per_user // classes_per_user
            # train_dset = build_shards_dataset(train_dset, samples_per_shard, classes_per_user, self.node_id)
        else:
            raise ValueError(
                "Unknown train label distribution: {}.".format(
                    config["train_label_distribution"]
                )
            )

        if self.dset.startswith("domainnet"):
            train_transform = T.Compose(
                [
                    T.RandomResizedCrop(32, scale=(0.75, 1)),
                    T.RandomHorizontalFlip(),
                    # T.ToTensor()
                ]
            )

            # Cache before transform to preserve transform randomness
            train_dset = TransformDataset(CacheDataset(train_dset), train_transform)

        self.classes_of_interest = classes

        val_prop = config.get("validation_prop", 0)
        val_dset = None
        if val_prop > 0:
            val_size = int(val_prop * len(train_dset))
            train_size = len(train_dset) - val_size
            train_dset, val_dset = torch.utils.data.random_split(
                train_dset, [train_size, val_size]
            )
            # self.val_dloader = DataLoader(val_dset, batch_size=batch_size*len(self.device_ids), shuffle=True)
            self.val_dloader = DataLoader(val_dset, batch_size=batch_size, shuffle=True)

        self.train_indices = train_indices
        # self.dloader = DataLoader(train_dset, batch_size=batch_size*len(self.device_ids), shuffle=True)
        self.dloader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)

        if config["test_label_distribution"] == "iid":
            pass
        # If non_iid, each users ge the whole test set for each of its
        # support classes
        elif config["test_label_distribution"] == "support":
            classes = config["support"][str(self.node_id)]
            test_dset, _ = filter_by_class(test_dset, classes)
        elif config["test_label_distribution"] == "non_iid":

            test_y, test_idx_split, _ = non_iid_balanced(
                self.dset_obj,
                len(users_with_same_dset),
                config["test_samples_per_user"],
                is_train=False,
            )

            train_indices = test_idx_split[self.node_id - 1]
            test_dset = Subset(test_dset, train_indices)
        else:
            raise ValueError(
                "Unknown test label distribution: {}.".format(
                    config["test_label_distribution"]
                )
            )

        if self.dset.startswith("domainnet"):
            test_dset = CacheDataset(test_dset)

        self._test_loader = DataLoader(test_dset, batch_size=batch_size)
        # TODO: fix print_data_summary
        # self.print_data_summary(train_dset, test_dset, val_dset=val_dset)

    def local_train(self, dataset, **kwargs):
        """
        Train the model locally
        """
        raise NotImplementedError

    def local_test(self, dataset, **kwargs):
        """
        Test the model locally
        """
        raise NotImplementedError

    def get_representation(self, **kwargs):
        """
        Share the model representation
        """
        raise NotImplementedError

    def run_protocol(self):
        raise NotImplementedError

    def print_data_summary(self, train_test, test_dset, val_dset=None):
        """
        Print the data summary
        """

        train_sample_per_class = {}
        i = 0
        for x, y in train_test:
            train_sample_per_class[y] = train_sample_per_class.get(y, 0) + 1
            print("train count: ", i)
            i += 1

        i = 0
        if val_dset is not None:
            val_sample_per_class = {}
            for x, y in val_dset:
                val_sample_per_class[y] = val_sample_per_class.get(y, 0) + 1
                print("val count: ", i)
                i += 1
        i = 0
        test_sample_per_class = {}
        for x, y in test_dset:
            test_sample_per_class[y] = test_sample_per_class.get(y, 0) + 1
            print("test count: ", i)
            i += 1

        print("Node: {} data distribution summary".format(self.node_id))
        print(type(train_sample_per_class.items()))
        print(
            "Train samples per class: {}".format(sorted(train_sample_per_class.items()))
        )
        print(
            "Train samples per class: {}".format(len(train_sample_per_class.items()))
        )
        if val_dset is not None:
            print(
                "Val samples per class: {}".format(len(val_sample_per_class.items()))
            )
        print(
            "Test samples per class: {}".format(len(test_sample_per_class.items()))
        )


class BaseServer(BaseNode):
    """
    Abstract class for orchestrator
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.num_users = config["num_users"]
        self.users = list(range(1, self.num_users + 1))
        self.set_data_parameters(config)

    def set_data_parameters(self, config):
        test_dset = self.dset_obj.test_dset
        batch_size = config["batch_size"]
        self._test_loader = DataLoader(test_dset, batch_size=batch_size)

    def aggregate(self, representation_list, **kwargs):
        """
        Aggregate the knowledge from the users
        """
        raise NotImplementedError

    def test(self, dataset, **kwargs):
        """
        Test the model on the server
        """
        raise NotImplementedError

    def get_model(self, **kwargs):
        """
        Get the model
        """
        raise NotImplementedError

    def run_protocol(self):
        raise NotImplementedError


class CommProtocol(object):
    """
    Communication protocol tags for the server and users
    """

    ROUND_START = 0  # Server signals the start of a round
    REPR_ADVERT = 1  # users advertise their representations with the server
    REPRS_SHARE = 2  # Server shares representations with users
    C_SELECTION = 3  # users send their selected collaborators to the server
    KNLDG_SHARE = 4  # Server shares selected knowledge with users
    ROUND_STATS = 5  # users send their stats to the server


class BaseFedAvgClient(BaseClient):
    """
    Abstract class for FedAvg based algorithms
    """

    def __init__(self, config, comm_protocol=CommProtocol) -> None:
        super().__init__(config)
        self.config = config
        self.model_save_path = "{}/saved_models/node_{}.pt".format(
            self.config["results_path"], self.node_id
        )
        self.tag = comm_protocol

        self.model_keys_to_ignore = []
        if not self.config.get(
            "average_last_layer", True
        ):  # By default include last layer
            keys = self.model_utils.get_last_layer_keys(self.get_model_weights())
            self.model_keys_to_ignore.extend(keys)

    def local_train(self, epochs):
        """
        Train the model locally
        """
        avg_loss, avg_acc = 0, 0
        for epoch in range(epochs):
            # if self.node_id ==1:
            #     tr_loss, tr_acc = self.model_utils.train(self.model, self.optim,
            #                                     self.dloader, self.loss_fn,
            #                                     self.device, self._test_loader)
            # else:
            tr_loss, tr_acc = self.model_utils.train(
                self.model, self.optim, self.dloader, self.loss_fn, self.device
            )

            avg_loss += tr_loss
            avg_acc += tr_acc

        avg_loss /= epochs
        avg_acc /= epochs

        return avg_loss, avg_acc

    def local_test(self, **kwargs):
        """
        Test the model locally, not to be used in the traditional FedAvg
        """
        test_loss, acc = self.model_utils.test(
            self.model, self._test_loader, self.loss_fn, self.device
        )
        if acc > self.best_acc:
            self.best_acc = acc
            self.model_utils.save_model(self.model, self.model_save_path)
        return acc

    def get_model_weights(self) -> OrderedDict[str, Tensor]:
        """
        Share the model weights (on the cpu)
        """
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def set_model_weights(self, model_wts: OrderedDict[str, Tensor], keys_to_ignore=[]):
        """
        Set the model weights
        """
        model_wts = copy.copy(model_wts)

        if len(keys_to_ignore) > 0:
            for key in keys_to_ignore:
                if key in model_wts.keys():
                    model_wts.pop(key)

        for key in model_wts.keys():
            model_wts[key] = model_wts[key].to(self.device)

        self.model.load_state_dict(model_wts, strict=len(keys_to_ignore) == 0)

    def weighted_aggregate(
        self,
        models_wts: Dict[int, OrderedDict[str, Tensor]],
        collab_weights_dict: Dict[int, float],
        keys_to_ignore=[],
        label_dict: Optional[Dict[int, Dict[str, int]]] = None,
    ):
        """
        Aggregate the model weights
        """

        selected_collab = [id for id, w in collab_weights_dict.items() if w > 0]

        first_model = models_wts[selected_collab[0]]

        models_coeffs = [
            (id, models_wts[id], collab_weights_dict[id]) for id in selected_collab
        ]

        last_layer_keys = []
        if label_dict is not None:
            last_layer_keys = [
                key
                for key in self.model_utils.get_last_layer_keys(
                    models_wts[self.node_id]
                )
            ]
            keys_to_ignore.extend(last_layer_keys)

        is_init = False
        agg_wts = OrderedDict()
        for _, model, coeff in models_coeffs:
            if coeff == 0:
                continue
            for key in first_model.keys():
                if key in keys_to_ignore:
                    continue
                if not is_init:
                    agg_wts[key] = coeff * model[key]
                else:
                    agg_wts[key] += coeff * model[key]
            is_init = True

        if label_dict is not None:
            last_layer_weight_key = [key for key in last_layer_keys if "weight" in key]
            last_layer_bias_key = [key for key in last_layer_keys if "bias" in key]
            if len(last_layer_weight_key) != 1 and len(last_layer_bias_key) != 1:
                raise ValueError(
                    "Unsupported last layer format, expected one weights layer and one bias."
                )

            last_layer_weight_key = last_layer_weight_key[0]
            last_layer_bias_key = last_layer_bias_key[0]

            # Host is the user's model
            # Ext are incoming models
            host_label_to_idx = label_dict[self.node_id]
            labels_coeff_sum = {
                host_idx: 0.0 for host_idx in label_dict[self.node_id].values()
            }

            # Match and aggregate weights of the last layer based on each
            # node's label dictionary
            host_model = models_wts[self.node_id]
            agg_wts[last_layer_weight_key] = torch.zeros_like(
                host_model[last_layer_weight_key]
            )
            for id, ext_model, coeff in models_coeffs:
                ext_last_layer = coeff * ext_model[last_layer_weight_key]
                ext_label_to_idx = label_dict[id]
                labels_map = [
                    (ext_idx, host_label_to_idx[ext_label])
                    for ext_label, ext_idx in ext_label_to_idx.items()
                    if ext_label in host_label_to_idx
                ]
                for ext_idx, host_idx in labels_map:
                    labels_coeff_sum[host_idx] += coeff
                    agg_wts[last_layer_weight_key][host_idx] += ext_last_layer[ext_idx]

            # Normalize the aggregated weights
            for label_idx, coeff_sum in labels_coeff_sum.items():
                if coeff_sum > 0:
                    agg_wts[last_layer_weight_key][label_idx] /= coeff_sum

            # TODO Bias aggregation ?
            agg_wts[last_layer_bias_key] = host_model[last_layer_bias_key]

        return agg_wts

    def get_collaborator_weights(
        self, reprs_dict: Dict[int, OrderedDict[int, Tensor]]
    ) -> Dict[int, float]:
        raise NotImplementedError

    def run_protocol(self):
        raise NotImplementedError


class BaseFedAvgServer(BaseServer):
    """
    Abstract class for orchestrator
    """

    def __init__(self, config, comm_protocol=CommProtocol) -> None:
        super().__init__(config)
        self.tag = comm_protocol

    def send_representations(self, representations, tag=None):
        for user_node in self.users:
            self.comm_utils.send_signal(
                dest=user_node,
                data=representations,
                tag=self.tag.REPRS_SHARE if tag is None else tag,
            )
