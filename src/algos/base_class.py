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
import time
import torch.utils.data
import gc

from utils.communication.comm_utils import CommunicationManager
from utils.plot_utils import PlotUtils
from utils.data_utils import (
    random_samples,
    filter_by_class,
    get_dataset,
    non_iid_balanced,
    balanced_subset,
    gia_client_dataset,
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
from utils.types import ConfigType, TorchModelType
from utils.dropout_utils import NodeDropout

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
        self.set_constants()
        self.config = config
        self.comm_utils = comm_utils
        self.node_id = self.comm_utils.get_rank()
        self.comm_utils.register_node(self)
        self.is_working = True

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

        dropout_seed = 1 * config.get("num_users", 9) + self.node_id * config.get("num_users", 9) + config.get("seed", 20) # arbitrarily chosen
        dropout_rng = random.Random(dropout_seed)
        self.dropout = NodeDropout(self.node_id, config["dropout_dicts"], dropout_rng)

        if "gia" in config and self.node_id in config["gia_attackers"]:
            self.gia_attacker = True
        
        self.log_memory = config.get("log_memory", False)

        self.stats : Dict[str, int | float | List[int]] = {}

        self.streaming_aggregation = config.get("streaming_aggregation", False)

    def set_constants(self) -> None:
        """Add docstring here"""
        self.best_acc = 0.0
        self.round = 0
        self.EMPTY_MODEL_TAG = "EMPTY_MODEL"

    def setup_logging(self, config: ConfigType) -> None:
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

        self.log_utils = LogUtils(config)
        if self.node_id == 0:
            self.log_utils.log_console("Config: {}".format(config))

    def setup_cuda(self, config: ConfigType) -> None:
        """add docstring here"""
        # Need a mapping from rank to device id
        if (config.get("assign_based_on_host", False)):
            device_ids_map = config["device_ids"]
            node_name = f"node_{self.node_id}"
            self.device_ids = device_ids_map[node_name]
        else:
            hostname_to_device_ids = config["hostname_to_device_ids"]
            hostname = os.uname().nodename
            # choose a random one of the available devices
            ind = self.node_id % len(hostname_to_device_ids[hostname])
            self.device_ids = [hostname_to_device_ids[hostname][ind]]
        gpu_id = self.device_ids[0]

        if isinstance(gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")
            print(f"Using GPU: cuda:{gpu_id}")
        elif gpu_id == "cpu":
            self.device = torch.device("cpu")
            print("Using CPU")
        else:
            # Fallback in case of no GPU availability
            self.device = torch.device("cpu")
            print("Using CPU (Fallback)")


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
        # if "gia" in config:
            # print("setting optim to gia")
            # optim = torch.optim.SGD
        num_classes = self.dset_obj.num_cls
        num_channels = self.dset_obj.num_channels
        self.model = self.model_utils.get_model(
            model_name=config["model"],
            dset=self.dset,
            device=self.device,
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

    def set_shared_exp_parameters(self, config: ConfigType) -> None:
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
                self.communities = get_dset_communities(config["num_users"], num_dset) # type: ignore
            elif community_type is None or number_of_communities == 1:
                all_users = list(range(1, config["num_users"] + 1)) # type: ignore
                self.communities = {user: all_users for user in all_users}
            elif community_type == "random":
                self.communities = get_random_communities(
                    config["num_users"], number_of_communities # type: ignore
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
        # if self.node_id == 0:
        #     self.log_utils.log_console(f"Communities: {self.communities}")

    def local_round_done(self) -> None:
        self.round += 1

    def get_model_weights(self, chop_model:bool=False) -> Dict[str, int|Dict[str, Any]]:
        """
        Share the model weights
        params:
        @chop_model: bool, if True, the model will only send the client part of the model. Only being used by Split Learning
        """
        if chop_model:
            model, _ = self.model_utils.get_split_model(self.model, self.config["split_layer"])
            model = model.state_dict()
        else:
            model = self.model.state_dict()
        message: Dict[str, int|Dict[str, Any]] = {"sender": self.node_id, "round": self.round, "model": model}

        if "gia" in self.config and hasattr(self, 'images') and hasattr(self, 'labels'):
            # also stream image and labels
            message["images"] = self.images
            message["labels"] = self.labels

        # Move to CPU before sending
        if isinstance(message["model"], dict):
            for key in message["model"].keys():
                message["model"][key] = message["model"][key].to("cpu")

        return message

    def get_local_rounds(self) -> int:
        return self.round

    @abstractmethod
    def run_protocol(self) -> None:
        """Add docstring here"""
        raise NotImplementedError

    def round_init(self) -> None:
        """
        Things to do at the start of each round.
        """
        self.round_start_time = time.time()
    
    def round_finalize(self) -> None:
        """
        Things to do at the end of each round.
        """
        self.round_end_time = time.time()
        self.round_duration = self.round_end_time - self.round_start_time

        self.stats["time_elapsed"] = self.stats.get("time_elapsed", 0) + self.round_duration # type: ignore
        
        self.stats["bytes_received"], self.stats["bytes_sent"] = self.comm_utils.get_comm_cost()

        self.stats["peak_dram"], self.stats["peak_gpu"] = self.get_memory_metrics()

        self.log_metrics(stats=self.stats, iteration=self.round)


    def log_metrics(self, stats: Dict[str, Any], iteration: int) -> None:
        """
        Centralized method to log metrics.

        Args:
            stats (Dict[str, Any]): Dictionary containing metric names and their values.
            iteration (int): Current iteration or round number.
        """
        # Log to console
        self.log_utils.log_console(
            f"Round {iteration} done for Node {self.node_id}, stats {stats}"
        )

        # Log scalar metrics to TensorBoard
        for key, value in stats.items():
            if isinstance(value, (float, int)):
                # Determine the category based on the key
                if "loss" in key.lower():
                    tb_key = f"{key}/loss"
                elif "acc" in key.lower() or "accuracy" in key.lower():
                    tb_key = f"{key}/accuracy"
                else:
                    tb_key = key  # Generic key

                self.log_utils.log_tb(key=tb_key, value=value, iteration=iteration)

        # Log numpy arrays if present
        for key, value in stats.items():
            if isinstance(value, np.ndarray):
                self.log_utils.log_npy(key=key, value=value)

        # Log all stats to CSV
        for key, value in stats.items():
            self.log_utils.log_csv(key=key, value=value, iteration=iteration)

        # Log images if present
        if "images" in stats:
            self.log_utils.log_image(
                imgs=stats["images"], key="sample_images", iteration=iteration
            )

    @abstractmethod
    def receive_and_aggregate(self):
        """Add docstring here"""
        raise NotImplementedError


    def strip_empty_models(self,  models_wts: List[OrderedDict[str, Any]],
        collab_weights: Optional[List[float]] = None) -> Any:
        repr_list = []
        if collab_weights is not None:
            weight_list = []
            for i, model_wts in enumerate(models_wts):
                if self.EMPTY_MODEL_TAG not in model_wts and collab_weights[i] > 0:
                    repr_list.append(model_wts)
                    weight_list.append(collab_weights[i])
            return repr_list, weight_list
        else:
            for model_wts in models_wts:
                if self.EMPTY_MODEL_TAG not in model_wts:
                    repr_list.append(model_wts)
            return repr_list, None

    def get_and_set_working(self, round: Optional[int] = None) -> bool:
        is_working = self.dropout.is_available()
        if not is_working:
            self.log_utils.log_console(
                f"Client {self.node_id} is not working {'in round ' if round else 'in this round'}."
            )
            self.comm_utils.set_is_working(False)
        else:
            self.comm_utils.set_is_working(True)
        return is_working

    def set_model_weights(
        self, model_wts: TorchModelType, keys_to_ignore: List[str] = []
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

    def push(self, neighbors: int | List[int]) -> None:
        """
        Pushes the model to the neighbors.
        """
        
        data_to_send = self.get_model_weights()

        # if it is a list, send to all neighbors
        if isinstance(neighbors, list):
            for neighbor in neighbors:
                self.comm_utils.send(neighbor, data_to_send)
        else:
            self.comm_utils.send(neighbors, data_to_send)

    def calculate_cpu_tensor_memory(self) -> int:
        total_memory = 0
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj.device.type == 'cpu': # type: ignore
                total_memory += obj.element_size() * obj.nelement()
        return total_memory

    def get_memory_metrics(self) -> Tuple[float | int, float | int]:
        """
        Get memory metrics
        """
        peak_dram, peak_gpu = 0, 0
        if self.log_memory:
            peak_dram = self.calculate_cpu_tensor_memory()
            peak_gpu = int(torch.cuda.max_memory_allocated()) # type: ignore
        return peak_dram, peak_gpu

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
        if "gia" in config: 
            if int(self.node_id) in self.config["gia_attackers"]:
                self.gia_attacker = True
            self.params_s = dict()
            self.params_t = dict()
            # Track neighbor updates with a dictionary mapping neighbor_id to their updates
            self.neighbor_updates = defaultdict(list)
            # Track which neighbors we've already attacked
            self.attacked_neighbors = set()

            self.base_params = [key for key, _ in self.model.named_parameters()]

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

        # after setting data loaders, save client dataset
        # TODO verify this .data and .labels fields are correct
        if "gia" in config:
            # Extract data and labels
            train_data = torch.stack([data[0] for data in self.train_dset])
            train_labels = torch.tensor([data[1] for data in self.train_dset])

            self.log_utils.log_gia_image(train_data, 
                                         train_labels,
                                         self.node_id)

    def set_data_parameters(self, config: ConfigType) -> None:

        # Train set and test set from original dataset
        train_dset = self.dset_obj.train_dset
        test_dset = self.dset_obj.test_dset

        # Handle GIA case first, before any other modifications
        if "gia" in config:
            # Select 10 random labels and exactly one image per label for both train and test
            train_dset, test_dset, classes, train_indices = gia_client_dataset(
                train_dset, test_dset, num_labels=10
            )
            
            assert len(train_dset) == 10, "GIA should have exactly 10 samples in train set"
            assert len(test_dset) == 10, "GIA should have exactly 10 samples in test set"
            
            # Store the images and labels in tensors, matching the format from your example
            self.images = []
            self.labels = []
            
            # Collect images and labels in order
            for idx in range(len(train_dset)):
                img, label = train_dset[idx]
                self.images.append(img)
                self.labels.append(torch.tensor([label]))
                
            # Stack/concatenate into final tensors
            self.images = torch.stack(self.images)  # Shape: [10, C, H, W]
            self.labels = torch.cat(self.labels)    # Shape: [10]
            
            # Set up the dataloaders with batch_size equal to dataset size for single-pass training
            self.classes_of_interest = classes
            self.train_indices = train_indices
            self.train_dset = train_dset
            self.dloader: DataLoader[Any] = DataLoader(train_dset, batch_size=len(train_dset), shuffle=False)
            self._test_loader: DataLoader[Any] = DataLoader(test_dset, batch_size=len(test_dset), shuffle=False)
            print("Using GIA data setup")
            print(self.labels)
        else:
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

            # reduce test_dset size
            if config.get("test_samples_per_user", 0) != 0:
                print(f"Reducing test size to {config.get('test_samples_per_user', 0)}")
                reduced_test_size = config.get("test_samples_per_user", 0)
                indices = np.random.choice(len(test_dset), reduced_test_size, replace=False)
                test_dset = Subset(test_dset, indices)
            print(f"test_dset size: {len(test_dset)}")

            self._test_loader = DataLoader(test_dset, batch_size=batch_size)
            # TODO: fix print_data_summary
            # self.print_data_summary(train_dset, test_dset, val_dset=val_dset)

    def local_train(self, round: int, epochs: int = 1, **kwargs: Any) -> Tuple[float, float, float]:
        """
        Train the model locally
        """
        start_time = time.time()

        self.is_working = self.get_and_set_working(round)

        if self.is_working:
            avg_loss, avg_acc = 0, 0
            for _ in range(epochs):
                tr_loss, tr_acc = self.model_utils.train(
                    self.model, self.optim, self.dloader, self.loss_fn, self.device, malicious_type=self.config.get("malicious_type", "normal"), config=self.config, node_id=self.node_id, gia=self.config.get("gia", False)
                )            
                avg_loss += tr_loss
                avg_acc += tr_acc
            avg_loss /= epochs
            avg_acc /= epochs
        else:
            avg_loss, avg_acc = float('nan'), float('nan')
            # sleep for a while to simulate the time taken for training
            time.sleep(2)
        end_time = time.time()
        time_taken = end_time - start_time

        self.log_utils.log_console(
            "Client {} finished training with loss {:.4f}, accuracy {:.4f}, time taken {:.2f} seconds".format(
                self.node_id, avg_loss, avg_acc, time_taken
            )
        )
        self.log_utils.log_summary(
            "Client {} finished training with loss {:.4f}, accuracy {:.4f}, time taken {:.2f} seconds".format(
                self.node_id, avg_loss, avg_acc, time_taken
            )
        )

        self.log_utils.log_tb(
            f"train_loss/client{self.node_id}", avg_loss, round
        )
        self.log_utils.log_tb(
            f"train_accuracy/client{self.node_id}", avg_acc, round
        )
 
        self.stats["train_loss"], self.stats["train_acc"], self.stats["train_time"] = avg_loss, avg_acc, time_taken

        return avg_loss, avg_acc, time_taken

    def local_test(self, **kwargs: Any) -> float | Tuple[float, float] | Tuple[float, float, float] | None:
        """
        Test the model locally
        """
        raise NotImplementedError

    def receive_and_aggregate(self):
        """
        Receive the model weights from the server and aggregate them
        """
        if self.is_working:
            repr = self.comm_utils.receive([self.server_node])[0]
            if "round" in repr:
                round = repr["round"]
            if "sender" in repr:
                sender = repr["sender"]
            assert "model" in repr, "Model not found in the received message"
            self.set_model_weights(repr["model"])

    def receive_attack_and_aggregate(self, neighbors: List[int], round: int, num_neighbors: int) -> None:
        """
        Receives updates, launches GIA attack when second update is seen from a neighbor
        """
        from utils.gias import gia_main
        
        if self.is_working:
            # Receive the model updates from the neighbors
            model_updates = self.comm_utils.receive(node_ids=neighbors)
            assert len(model_updates) == num_neighbors

            for neighbor_info in model_updates:
                neighbor_id = neighbor_info["sender"]
                neighbor_model = neighbor_info["model"]
                neighbor_model = OrderedDict(
                    (key, value) for key, value in neighbor_model.items()
                    if key in self.base_params
                )

                neighbor_images = neighbor_info["images"]
                neighbor_labels = neighbor_info["labels"]

                # Store this update
                self.neighbor_updates[neighbor_id].append({
                    "model": neighbor_model,
                    "images": neighbor_images,
                    "labels": neighbor_labels
                })

                # Check if we have 2 updates from this neighbor and haven't attacked them yet
                if len(self.neighbor_updates[neighbor_id]) == 2 and neighbor_id not in self.attacked_neighbors:
                    print(f"Client {self.node_id} attacking {neighbor_id}!")
                    
                    # Get the two parameter sets for the attack
                    p_s = self.neighbor_updates[neighbor_id][0]["model"]
                    p_t = self.neighbor_updates[neighbor_id][1]["model"]
                    
                    # Launch the attack
                    if result := gia_main(p_s, 
                                        p_t, 
                                        self.base_params, 
                                        self.model, 
                                        neighbor_labels, 
                                        neighbor_images, 
                                        self.node_id):
                        output, stats = result
                        
                        # log output and stats as image
                        self.log_utils.log_gia_image(output, neighbor_labels, neighbor_id, label=f"round_{round}_reconstruction")
                        self.log_utils.log_summary(f"round {round} gia targeting {neighbor_id} stats: {stats}")
                    else:
                        self.log_utils.log_summary(f"Client {self.node_id} failed to attack {neighbor_id} in round {round}!")
                        print(f"Client {self.node_id} failed to attack {neighbor_id}!")
                        continue
                    
                    # Mark this neighbor as attacked
                    self.attacked_neighbors.add(neighbor_id)
                    
                    # Optionally, clear the stored updates to save memory
                    del self.neighbor_updates[neighbor_id]

            self.aggregate(model_updates, keys_to_ignore=self.model_keys_to_ignore)

    def receive_pushed_and_aggregate(self, remove_multi = True) -> None:
        model_updates = self.comm_utils.receive_pushed()

        if len(model_updates) > 0:
            if self.is_working:
                # Remove multiple models of different rounds from each node
                if remove_multi:
                    to_aggregate = {}
                    for model in model_updates:
                        sender = model.get("sender", 0)
                        if sender not in to_aggregate or to_aggregate[sender].get("round", 0) < model.get("round", 0):
                            to_aggregate[sender] = model
                    model_updates = list(to_aggregate.values())
                # Aggregate the representations
                repr = model_updates[0]
                assert "model" in repr, "Model not found in the received message"
                self.set_model_weights(repr["model"])
        else:
            print("No one pushed model updates for this round.")


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
        if "gia" not in config:
            self._test_loader = DataLoader(test_dset, batch_size=batch_size)
        else:
            _, test_data, labels, indices = gia_client_dataset(self.dset_obj.train_dset, test_dset)
            self._test_loader = DataLoader(test_data, batch_size=10)
    def aggregate(
        self, representation_list: List[OrderedDict[str, Any]], **kwargs: Any
    ) -> OrderedDict[str, Tensor]:
        """
        Aggregate the knowledge from the users
        """
        raise NotImplementedError

    def test(self, **kwargs: Any) -> Any:
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


    def local_test(self, **kwargs: Any) -> Tuple[float, float]:
        """
        Test the model locally, not to be used in the traditional FedAvg
        """
        start_time = time.time()
        test_loss, acc = self.model_utils.test(
            self.model, self._test_loader, self.loss_fn, self.device
        )
        end_time = time.time()
        time_taken = end_time - start_time
        if acc > self.best_acc:
            self.best_acc = acc
            self.model_utils.save_model(self.model, self.model_save_path)
        
        self.stats["test_loss"], self.stats["test_acc"], self.stats["test_time"] = test_loss, acc, time_taken

        return test_loss, acc

    def aggregate(
        self,
        models_wts: List[OrderedDict[str, Any]],
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

        # Handle dropouts and re-normalize the weights
        models_wts, collab_weights = self.strip_empty_models(models_wts, collab_weights)
        collab_weights = [w / sum(collab_weights) for w in collab_weights]

        senders = [model["sender"] for model in models_wts if "sender" in model]
        rounds = [model["round"] for model in models_wts if "round" in model]
        for i in range(len(models_wts)):
            assert "model" in models_wts[i], "Model not found in the received message"
            models_wts[i] = models_wts[i]["model"]

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
        
        self.set_model_weights(agg_wts)
        return None

    def aggregate_streaming(
            self,
            agg_wts: OrderedDict[str, Tensor],
            model_wts: OrderedDict[str, Tensor],
            coeff: float,
            is_initialized: bool,
            keys_to_ignore: List[str],
        ) -> None:
        """
        Incrementally aggregates the model weights into the aggregation state.

        Args:
            agg_wts (OrderedDict[str, Tensor]): Aggregated weights (to be updated in place).
            model_wts (OrderedDict[str, Tensor]): Weights of the current model to aggregate.
            coeff (float): Collaboration weight for the current model.
            is_initialized (bool): Whether the aggregation state is initialized.
            keys_to_ignore (List[str]): Keys to ignore during aggregation.

        Returns:
            None
        """
        for key in self.model.state_dict().keys():
            if key in keys_to_ignore:
                continue
            if not is_initialized:
                # Initialize the aggregation state
                agg_wts[key] = coeff * model_wts[key].to(self.device)
            else:
                # Incrementally update the aggregation state
                agg_wts[key] += coeff * model_wts[key].to(self.device)

        return None

    def receive_pushed_and_aggregate(self, remove_multi = True) -> None:
        model_updates = self.comm_utils.receive_pushed()
        if self.is_working:
            # Remove multiple models of different rounds from each node
            if remove_multi:
                to_aggregate = {}
                for model in model_updates:
                    sender = model.get("sender", 0)
                    if sender not in to_aggregate or to_aggregate[sender].get("round", 0) < model.get("round", 0):
                        to_aggregate[sender] = model
                model_updates = list(to_aggregate.values())
            # Aggregate the representations
            self.aggregate(model_updates, keys_to_ignore=self.model_keys_to_ignore)

    def receive_and_aggregate_streaming(self, neighbors: List[int]) -> None:
        if self.is_working:
            # Initialize the aggregation state
            agg_wts: OrderedDict[str, Tensor] = OrderedDict()
            is_initialized = False
            total_weight = 0.0  # To re-normalize weights after handling dropouts

            # Include the current node's model in the aggregation
            current_model_wts = self.get_model_weights()
            assert "model" in current_model_wts, "Model not found in the current model."
            current_model_wts = current_model_wts["model"]
            current_weight = 1.0 / (len(neighbors) + 1)  # Weight for the current node
            self.aggregate_streaming(
                agg_wts,
                current_model_wts,
                coeff=current_weight,
                is_initialized=is_initialized,
                keys_to_ignore=self.model_keys_to_ignore,
            )
            is_initialized = True
            total_weight += current_weight

            # Process models from neighbors one at a time
            for neighbor in neighbors:
                # Receive the model update from the current neighbor
                model_update = self.comm_utils.receive(node_ids=[neighbor])
                model_update, _ = self.strip_empty_models(model_update)
                if len(model_update) == 0:
                    # Skip empty models (dropouts)
                    continue
                
                model_update = model_update[0]
                assert "model" in model_update, "Model not found in the received message"
                model_wts = model_update["model"]

                # Get the collaboration weight for the current neighbor
                coeff = current_weight # Default weight

                # Perform streaming aggregation for the current model
                self.aggregate_streaming(
                    agg_wts,
                    model_wts,
                    coeff=coeff,
                    is_initialized=is_initialized,
                    keys_to_ignore=self.model_keys_to_ignore,
                )
                total_weight += coeff

            # Re-normalize the aggregated weights if there were dropouts
            if total_weight > 0:
                for key in agg_wts.keys():
                    agg_wts[key] /= total_weight

            # Update the model with the aggregated weights
            self.set_model_weights(agg_wts)

    def receive_and_aggregate(self, neighbors: List[int]) -> None:
        if hasattr(self, "gia_attacker"):
            self.receive_attack_and_aggregate(neighbors, it, len(neighbors))
        if self.streaming_aggregation:
            self.receive_and_aggregate_streaming(neighbors)
        else:
            if self.is_working:
                # Receive the model updates from the neighbors
                model_updates = self.comm_utils.receive(node_ids=neighbors)
                # Aggregate the representations
                self.aggregate(model_updates, keys_to_ignore=self.model_keys_to_ignore)

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
