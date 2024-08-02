import argparse
import grpc
import comm_pb2
import comm_pb2_grpc
import torch
import numpy as np
import sys
import os
import json
import collections
import time
import random

BASE_DIR = "/projectnb/ds598/projects/xthomas/temp/sonar"
sys.path.append(f"{BASE_DIR}/src/")
from torch.utils.data import DataLoader, Subset
from utils.data_utils import get_dataset
from utils.model_utils import ModelUtils
from utils.log_utils import LogUtils
from grpc_utils import deserialize_model, serialize_model
import hparams_registry

TEMP_TOTAL_NODES = 2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_client(args: argparse.Namespace):
    # Assign parsed args
    hostname = args.host
    client_id = args.client_id
    output_dir = args.output_dir
    device_offset = 1  # so that we don't run things on 0th gpu that is usually crowded

    dpath = f"{BASE_DIR}/data"
    model_utils = ModelUtils()
    loss_fn = torch.nn.CrossEntropyLoss()

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(
            args.algorithm,
            args.dataset,
            args.hparams_seed,  # To be used as seed in random_hparams
        )
    if args.hparams:
        hparams.update(json.loads(args.hparams))


    set_seed(args.trial_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Connect to server and send local average
    with grpc.insecure_channel(
        hostname,
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),  # 50MB
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),  # 50MB
        ],
    ) as channel:
        stub = comm_pb2_grpc.CommunicationServerStub(channel)
        user_id = stub.GetID(comm_pb2.Empty())
        config = {}
        config["log_path"] = output_dir
        os.makedirs(config["log_path"], exist_ok=True)
        config["load_existing"] = True # TODO: False before
        log_utils = LogUtils(config)
        log_utils.log_console(f"User got ID: {user_id.id}, Number: {user_id.num}")

        # Assign node_id
        node_id = user_id.num % TEMP_TOTAL_NODES
        device = f"cuda:{node_id + device_offset}"
        log_utils.log_console(f"Assigned Node ID: {node_id}, to GPU: {device}")

        dset_obj = get_dataset(args.dataset, dpath=dpath)
        org_train_dset = dset_obj.train_dset
        indices = np.random.permutation(len(org_train_dset))
        samples_per_client = 10000
        train_indices = indices[
            node_id * samples_per_client : (node_id + 1) * samples_per_client
        ]
        train_dset = Subset(org_train_dset, train_indices)
        dloader = DataLoader(train_dset, batch_size=hparams["batch_size"], shuffle=True)
        test_dset = dset_obj.test_dset
        test_loader = DataLoader(test_dset, batch_size=hparams["batch_size"])

        log_utils.log_console(
            f"Number of training samples for this client: {len(train_indices)}"
        )
        log_utils.log_console(
            f"Total number of training samples in original dataset: {len(org_train_dset)}"
        )

        model = model_utils.get_model(
            args.model_arch, args.dataset, device, [hparams["dropout_rate"]]
        )
        te_loss, te_acc = model_utils.test(model, test_loader, loss_fn, device)
        log_utils.log_console(
            f"Initial Test Loss: {te_loss:.3f}, Test Acc: {te_acc:.3f}"
        )
        log_utils.log_tb("test_loss", te_loss, 0)
        log_utils.log_tb("test_acc", te_acc, 0)
        optim = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"])

        g_model = stub.GetModel(comm_pb2.ID(id=user_id.id, num=user_id.num))
        g_model = deserialize_model(g_model.buffer, torch.device(device))
        model.load_state_dict(g_model, strict=False)
        log_utils.log_console("Received model from server")

        # Initialize logging metrics
        results = collections.defaultdict(lambda: [])

        results["client_id"] = client_id
        results["node_id"] = node_id
        results["train_indices"] = train_indices.tolist()
        results["percentage_samples"] = len(train_indices) / len(org_train_dset)
        results["initial_test_loss"] = te_loss
        results["initial_test_acc"] = te_acc

        # Add additional information to results
        results["dataset"] = args.dataset
        results["model_arch"] = args.model_arch
        results["algorithm"] = args.algorithm
        results["hparams_seed"] = args.hparams_seed
        results["trial_seed"] = args.trial_seed

        for k, v in sorted(hparams.items()):
            results[k] = v

        for i in range(1, 51):
            step_start_time = time.time()
            tr_loss, tr_acc = model_utils.train(model, optim, dloader, loss_fn, device)
            log_utils.log_console(
                f"Epoch {i}, Node {user_id.num}, Train Loss: {tr_loss:.3f}, Train Acc: {tr_acc:.3f}"
            )

            # Send model to server
            model_bytes = serialize_model(model.state_dict())

            stub.SendMessage(
                comm_pb2.Message(
                    model=comm_pb2.Model(buffer=model_bytes),
                    id=user_id.id,
                )
            )

            # Get global model from server
            g_model = stub.GetModel(comm_pb2.ID(id=user_id.id, num=user_id.num))
            g_model = deserialize_model(g_model.buffer, torch.device(device))

            # Load global model
            model.load_state_dict(g_model, strict=False)

            # Test global model
            te_loss, te_acc = model_utils.test(model, test_loader, loss_fn, device)
            log_utils.log_console(
                f"Test Loss after epoch {i}: {te_loss:.3f}, Test Acc: {te_acc:.3f}"
            )
            log_utils.log_tb("train_loss", tr_loss, i)
            log_utils.log_tb("train_acc", tr_acc, i)
            log_utils.log_tb("test_loss", te_loss, i)
            log_utils.log_tb("test_acc", te_acc, i)

            # Log metrics
            results["train_loss"].append(tr_loss)
            results["train_acc"].append(tr_acc)
            results["test_loss"].append(te_loss)
            results["test_acc"].append(te_acc)
            results["step_time"].append(time.time() - step_start_time)

        stub.SendBye(comm_pb2.ID(id=user_id.id))
        log_utils.log_console(
            f"Exiting... Node {user_id.num}, Final Test Acc: {te_acc:.3f}"
        )

        # Save results to file
        results_path = os.path.join(output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

        # Create done file
        with open(os.path.join(output_dir, "done"), "w") as f:
            f.write("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--model_arch", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--client_id", type=str, required=True)
    parser.add_argument("--hparams", type=str, help="JSON-serialized hparams dict")
    parser.add_argument(
        "--hparams_seed",
        type=int,
        default=0,
        help='Seed for random hparams (0 means "default hparams")',
    )
    parser.add_argument(
        "--trial_seed",
        type=int,
        default=0,
        help="Trial number (used for seeding split_dataset and " "random_hparams).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for everything else")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--algorithm", type=str, default="fedavg")
    args = parser.parse_args()
    run_client(args)
