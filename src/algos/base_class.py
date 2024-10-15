"""Add docstring here"""

from abc import ABC, abstractmethod
import sys
import torch
import numpy as np
import torch.utils
from torch.utils.data import DataLoader, Subset

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor
import copy
import random
import torch.utils.data

from utils.communication.comm_utils import CommunicationManager
from utils.plot_utils import PlotUtils
from utils.data_utils import (
    random_samples,
    filter_by_class,
    get_dataset,
    non_iid_balanced,
    balanced_subset,
    CacheDataset,
    TransformDataset,
    CorruptDataset,
)
from utils.log_utils import LogUtils
from utils.model_utils import ModelUtils
from utils.community_utils import (
    get_random_communities,
    get_dset_balanced_communities,
    get_dset_communities,
)
from utils.types import ConfigType

import torchvision.transforms as T  # type: ignore
import os

from yolo import YOLOLoss


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)  # type: ignore
    random.seed(seed)
    np.random.seed(seed)


class BaseNode(ABC):
    """BaseNode is an abstract base class that provides foundational functionalities for nodes in a distributed system. It handles configuration, logging, CUDA setup, model parameter settings, and shared experiment parameters.

    Attributes:
        comm_utils (CommunicationManager): Utility for communication management.
        node_id (int): Unique identifier for the node.
        dset (str): Dataset identifier for the node.
        device (torch.device): Device (CPU/GPU) to be used for computation.
        model_utils (ModelUtils): Utility for model-related operations.
        dset_obj (Dataset): Dataset object for the node.
        best_acc (float): Best accuracy achieved by the node.
        plot_utils (PlotUtils): Utility for plotting.
        log_utils (LogUtils): Utility for logging.
        device_ids (List[int]): List of device IDs for CUDA.
        model (torch.nn.Module): Model used by the node.
        optim (torch.optim.Optimizer): Optimizer for the model.
        loss_fn (torch.nn.Module): Loss function for the model.
        num_collaborators (int): Number of collaborators in the experiment.
        communities (Dict[int, List[int]]): Mapping of users to their communities.

    Methods:
        __init__(self, config: Dict[str, Any], comm_utils: CommunicationManager) -> None:
            Initializes the BaseNode with the given configuration and communication utilities.

        set_constants(self) -> None:
            Sets constant attributes for the node.

        setup_logging(self, config: Dict[str, ConfigType]) -> None:

        setup_cuda(self, config: Dict[str, ConfigType]) -> None:
            Sets up CUDA devices for the node based on the configuration.

        set_model_parameters(self, config: Dict[str, Any]) -> None:
            Sets model-related parameters including the model, optimizer, and loss function.

        set_shared_exp_parameters(self, config: Dict[str, ConfigType]) -> None:
            Sets shared experiment parameters including the number of collaborators and community settings.

        run_protocol(self) -> None:
            Abstract method to be implemented by subclasses, defining the protocol to be run by the node.
    """
    def __init__(
        self, config: Dict[str, Any], comm_utils: CommunicationManager
    ) -> None:
        self.comm_utils = comm_utils
        self.node_id = self.comm_utils.get_rank()
        self.comm_utils.register_node(self)

        self.setup_logging(config)

        # Support user specific dataset
        if isinstance(config["dset"], dict):
            if self.node_id != 0:
                config["dset"].pop("0") # type: ignore
            self.dset = str(config["dset"][str(self.node_id)]) # type: ignore
            config["dpath"] = config["dpath"][self.dset]
        else:
            self.dset = config["dset"]

        self.setup_cuda(config)
        self.model_utils = ModelUtils(self.device, config)

        self.dset_obj = get_dataset(self.dset, dpath=config["dpath"])
        self.set_constants()

    def set_constants(self) -> None:
        """Add docstring here"""
        self.best_acc = 0.0
        self.round = 0

    def setup_logging(self, config: Dict[str, ConfigType]) -> None:
        """
        Sets up logging for the node by creating necessary directories and initializing logging utilities.

        Args:
            config (Dict[str, ConfigType]): Configuration dictionary containing logging and plotting paths.

        Raises:
            SystemExit: If the log directory for the node already exists to prevent accidental overwrite.

        Side Effects:
            - Creates a log directory specific to the node.
            - Initializes PlotUtils and LogUtils with the given configuration.
            - Logs the configuration to the console if the node ID is 0.
        """
        try:
            config["log_path"] = f"{config['log_path']}/node_{self.node_id}" # type: ignore
            os.makedirs(config["log_path"])
        except FileExistsError:
            color_code = "\033[91m"  # Red color
            reset_code = "\033[0m"  # Reset to default color
            print(
                f"{color_code}Log directory for the node {self.node_id} already exists in {config['log_path']}"
            )
            print(f"Exiting to prevent accidental overwrite{reset_code}")
            sys.exit(1)

        # TODO: Check if the plot directory should be unique to each node
        try:
            self.plot_utils = PlotUtils(config)
        except FileExistsError:
            print(f"Plot directory for the node {self.node_id} already exists")

        self.log_utils = LogUtils(config)
        if self.node_id == 0:
            self.log_utils.log_console("Config: {}".format(config))

    def setup_cuda(self, config: Dict[str, ConfigType]) -> None:
        """add docstring here"""
        # Need a mapping from rank to device id
        device_ids_map = config["device_ids"]
        node_name = f"node_{self.node_id}"
        self.device_ids = device_ids_map[node_name]
        gpu_id = self.device_ids[0]

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")
            print(f"Using GPU: cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

    def set_model_parameters(self, config: Dict[str, Any]) -> None:
        # Model related parameters
        """Add docstring here"""
        optim_name = config.get("optimizer", "adam")
        if optim_name == "adam":
            optim = torch.optim.Adam
        elif optim_name == "sgd":
            optim = torch.optim.SGD
        else:
            raise ValueError(f"Unknown optimizer: {optim_name}.")
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
        if config.get("dset") == "pascal":
            self.loss_fn = YOLOLoss()
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()

    def set_shared_exp_parameters(self, config: Dict[str, ConfigType]) -> None:
        self.num_collaborators: int = config["num_collaborators"] # type: ignore
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
                raise ValueError(f"Unknown community type: {community_type}.")
        if self.node_id == 0:
            self.log_utils.log_console(f"Communities: {self.communities}")

    def local_round_done(self) -> None:
        self.round += 1

    def get_model_weights(self) -> Dict[str, Tensor]:
        """
        Share the model weights
        """
        return self.model.state_dict()

    def get_local_rounds(self) -> int:
        return self.round

    @abstractmethod
    def run_protocol(self) -> None:
        """Add docstring here"""
        raise NotImplementedError


class BaseClient(BaseNode):
    """
    Abstract class for all algorithms
    """

    def __init__(
        self, config: Dict[str, Any], comm_utils: CommunicationManager
    ) -> None:
        """Add docstring here"""
        super().__init__(config, comm_utils)
        self.server_node = 0
        self.set_parameters(config)

    def set_parameters(self, config: Dict[str, Any]) -> None:
        """
        Set the parameters for the user
        """

        # Set same seed for all users for data distribution and shared
        # parameters so that all users have the same data splits and shared
        # params
        seed = config["seed"]
        set_seed(seed)
        self.set_model_parameters(config)
        self.set_shared_exp_parameters(config)
        self.set_data_parameters(config)

    def set_data_parameters(self, config: ConfigType) -> None:

        # Train set and test set from original dataset
        train_dset = self.dset_obj.train_dset
        test_dset = self.dset_obj.test_dset

        # print("num train", len(train_dset))
        # print("num test", len(test_dset))

        if config.get("test_samples_per_class", None) is not None:
            test_dset, _ = balanced_subset(test_dset, config["test_samples_per_class"])

        samples_per_user = config["samples_per_user"]
        batch_size: int = config["batch_size"] # type: ignore
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
            indices = np.random.permutation(len(train_dset))
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
            classes = np.unique(train_y[user_idx]).tolist()
            # One plot per dataset
            # if user_idx == 0:
            #     print("using non_iid_balanced", alpha)
            #     self.plot_utils.plot_training_distribution(train_y,
            # self.dset, users_with_same_dset)
        elif config["train_label_distribution"] == "shard":
            raise NotImplementedError
            # classes_per_user = config["shards"]["classes_per_user"]
            # samples_per_shard = samples_per_user // classes_per_user
            # train_dset = build_shards_dataset(train_dset, samples_per_shard,
            # classes_per_user, self.node_id)
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

        if config.get("malicious_type", None) == "corrupt_data":
            corruption_fn_name = config.get("corruption_fn", "gaussian_noise")
            severity = config.get("corrupt_severity", 1)
            train_dset = CorruptDataset(CacheDataset(train_dset), corruption_fn_name, severity)
            print("created train dataset with corruption function: ", corruption_fn_name)

        self.classes_of_interest = classes

        val_prop = config.get("validation_prop", 0)
        val_dset = None
        if val_prop > 0:
            val_size = int(val_prop * len(train_dset))
            train_size = len(train_dset) - val_size
            train_dset, val_dset = torch.utils.data.random_split(
                train_dset, [train_size, val_size]
            )
            # self.val_dloader = DataLoader(val_dset, batch_size=batch_size*len(self.device_ids),
            # shuffle=True)
            self.val_dloader = DataLoader(val_dset, batch_size=batch_size, shuffle=True)

        assert isinstance(train_dset, torch.utils.data.Dataset), "train_dset must be a Dataset"
        self.train_indices = train_indices
        self.train_dset = train_dset
        self.dloader = DataLoader(train_dset, batch_size=batch_size, shuffle=True) # type: ignore

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

    def local_train(self, round: int, **kwargs: Any) -> None:
        """
        Train the model locally
        """
        raise NotImplementedError

    def local_test(self, **kwargs: Any) -> float | Tuple[float, float] | None:
        """
        Test the model locally
        """
        raise NotImplementedError

    def run_protocol(self) -> None:
        raise NotImplementedError

    def print_data_summary(
        self, train_test: Any, test_dset: Any, val_dset: Optional[Any] = None
    ) -> None:
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

        # print("Node: {} data distribution summary".format(self.node_id))
        # print(
        #     "Train samples per class: {}".format(sorted(train_sample_per_class.items()))
        # )
        # if val_dset is not None:
        #     print(
        #         "Val samples per class: {}".format(sorted(val_sample_per_class.items()))
        #     )
        # print(
        #     "Test samples per class: {}".format(sorted(test_sample_per_class.items()))
        # )


class BaseServer(BaseNode):
    """
    Abstract class for orchestrator
    """

    def __init__(
        self, config: Dict[str, Any], comm_utils: CommunicationManager
    ) -> None:
        """Add docstring here"""
        super().__init__(config, comm_utils)
        self.num_users = config["num_users"]
        self.users = list(range(1, self.num_users + 1))
        self.set_data_parameters(config)

    def set_data_parameters(self, config: Dict[str, Any]) -> None:
        """Add docstring here"""
        test_dset = self.dset_obj.test_dset
        batch_size = config["batch_size"]
        self._test_loader = DataLoader(test_dset, batch_size=batch_size)

    def aggregate(
        self, representation_list: List[OrderedDict[str, Tensor]], **kwargs: Any
    ) -> OrderedDict[str, Tensor]:
        """
        Aggregate the knowledge from the users
        """
        raise NotImplementedError

    def test(self, **kwargs: Any) -> List[float]:
        """
        Test the model on the server
        """
        raise NotImplementedError

    def get_model(self, **kwargs: Any) -> Any:
        """
        Get the model
        """
        raise NotImplementedError

    def run_protocol(self) -> None:
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

    def __init__(
        self,
        config: Dict[str, Any],
        comm_utils: CommunicationManager,
        comm_protocol: type[CommProtocol] = CommProtocol,
    ) -> None:
        """Add docstring here"""
        super().__init__(config, comm_utils)
        self.config = config
        self.model_save_path = f"{self.config['results_path']}/saved_models/node_{self.node_id}.pt"
        self.tag = comm_protocol

        self.model_keys_to_ignore = []
        if not self.config.get(
            "average_last_layer", True
        ):  # By default include last layer
            keys = self.model_utils.get_last_layer_keys(self.get_model_weights())
            self.model_keys_to_ignore.extend(keys)
 
    def local_train(self, epochs: int) -> Tuple[float, float]:
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

    def local_test(self, **kwargs: Any) -> Tuple[float, float]:
        """
        Test the model locally, not to be used in the traditional FedAvg
        """
        test_loss, acc = self.model_utils.test(
            self.model, self._test_loader, self.loss_fn, self.device
        )
        if acc > self.best_acc:
            self.best_acc = acc
            self.model_utils.save_model(self.model, self.model_save_path)
        return test_loss, acc

    def get_model_weights(self) -> OrderedDict[str, Tensor]:
        """
        Share the model weights (on the cpu)
        """
        return OrderedDict({k: v.cpu() for k, v in self.model.state_dict().items()})

    def set_model_weights(
        self, model_wts: OrderedDict[str, Tensor], keys_to_ignore: List[str] = []
    ) -> None:
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

    def aggregate(
        self,
        models_wts: List[OrderedDict[str, Tensor]],
        collab_weights: Optional[List[float]] = None,
        keys_to_ignore: List[str] = [],
    ) -> None:
        """ Aggregate the model weights using the collab_weights and then updates its own model weights
        If the collab_weights are not provided, then equal weights are assumed
        Args:
            models_wts (Dict[int, OrderedDict[str, Tensor]]): A dictionary where the key is the model ID and the value is an 
                ordered dictionary of model weights.
            collab_weights (Optional[List[float]]): A list of weights for each model. If not provided, equal weights are assumed.
            keys_to_ignore (List[str]): A list of keys to ignore during the aggregation process.

        Returns:
            None
        """
        models_coeffs: List[Tuple[OrderedDict[str, Tensor], float]] = []
        # insert the current model weights at the position self.node_id
        models_wts.insert(self.node_id - 1, self.get_model_weights())
        if collab_weights is None:
            collab_weights = [1.0 / len(models_wts) for _ in models_wts]
        for idx, model_wts in enumerate(models_wts):
            models_coeffs.append((model_wts, collab_weights[idx]))

        is_init = False
        agg_wts: OrderedDict[str, Tensor] = OrderedDict()
        for model, coeff in models_coeffs:
            for key in self.model.state_dict().keys():
                if key in keys_to_ignore:
                    continue
                if not is_init:
                    agg_wts[key] = coeff * model[key].to(self.device)
                else:
                    agg_wts[key] += coeff * model[key].to(self.device)
            is_init = True
        self.model.load_state_dict(agg_wts)
        return None

    def get_collaborator_weights(
        self, reprs_dict: Dict[int, OrderedDict[int, Tensor]]
    ) -> Dict[int, float]:
        """Add docstring here"""
        raise NotImplementedError

    def run_protocol(self) -> None:
        """Add docstring here"""
        raise NotImplementedError


class BaseFedAvgServer(BaseServer):
    """
    Abstract class for orchestrator
    """

    def __init__(
        self,
        config: Dict[str, Any],
        comm_utils: CommunicationManager,
        comm_protocol: type[CommProtocol] = CommProtocol,
    ) -> None:
        super().__init__(config, comm_utils)
        self.tag = comm_protocol

    def send_representations(
        self, representations: Dict[int, OrderedDict[str, Tensor]]
    ):
        self.comm_utils.broadcast(representations)
