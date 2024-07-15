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

BASE_DIR = "/projectnb/ds598/projects/xthomas/temp/sonar"
sys.path.append(f"{BASE_DIR}/src/")
from torch.utils.data import DataLoader, Subset
from utils.data_utils import get_dataset
from utils.model_utils import ModelUtils
from utils.log_utils import LogUtils
from grpc_utils import deserialize_model, serialize_model


def run_client(args: argparse.Namespace):
    # Assign parsed args
    dset = args.dataset
    hostname = args.host
    model_arch = args.model_arch
    client_id = args.client_id
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    dropout_rate = args.dropout_rate
    hparams_seed = args.hparams_seed
    trial_seed = args.trial_seed
    output_dir = args.output_dir
    gpu_index = args.gpu_index
    algorithm = args.algorithm

    # Compute local average
    device = f"cuda:{gpu_index}"  # Client GPU
    dpath = f"{BASE_DIR}/data"
    model_utils = ModelUtils()
    loss_fn = torch.nn.CrossEntropyLoss()

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
        config["load_existing"] = False
        log_utils = LogUtils(config)
        log_utils.log_console(f"User got ID: {user_id.id}, Number: {user_id.num}")
        node_id = user_id.num % 2
        log_utils.log_console(f"Assigned Node ID: {node_id}")
        log_utils.log_console(f"GPU Device: {device}")

        dset_obj = get_dataset(dset, dpath=dpath)
        org_train_dset = dset_obj.train_dset
        indices = np.random.permutation(len(org_train_dset))
        samples_per_client = 1000
        train_indices = indices[
            node_id * samples_per_client : (node_id + 1) * samples_per_client
        ]
        train_dset = Subset(org_train_dset, train_indices)
        dloader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
        test_dset = dset_obj.test_dset
        test_loader = DataLoader(test_dset, batch_size=batch_size)

        log_utils.log_console(
            f"Number of training samples for this client: {len(train_indices)}"
        )
        log_utils.log_console(
            f"Total number of training samples in original dataset: {len(org_train_dset)}"
        )

        model = model_utils.get_model(model_arch, dset, device, [dropout_rate])
        te_loss, te_acc = model_utils.test(model, test_loader, loss_fn, device)
        log_utils.log_console(
            f"Initial Test Loss: {te_loss:.3f}, Test Acc: {te_acc:.3f}"
        )
        log_utils.log_tb("test_loss", te_loss, 0)
        log_utils.log_tb("test_acc", te_acc, 0)
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

        g_model = stub.GetModel(comm_pb2.ID(id=user_id.id, num=user_id.num))
        g_model = deserialize_model(g_model.buffer, torch.device(device))
        model.load_state_dict(g_model, strict=False)
        log_utils.log_console("Received model from server")

        # Initialize logging metrics
        results = collections.defaultdict(lambda: [])

        results["client_id"] = client_id
        results["gpu_index"] = gpu_index
        results["node_id"] = node_id
        results["train_indices"] = train_indices.tolist()
        results["percentage_samples"] = len(train_indices) / len(org_train_dset)
        results["initial_test_loss"] = te_loss
        results["initial_test_acc"] = te_acc

        # Add additional information to results
        results["dataset"] = dset
        results["model_arch"] = model_arch
        results["algorithm"] = algorithm
        results["hparams_seed"] = hparams_seed
        results["trial_seed"] = trial_seed
        results["learning_rate"] = learning_rate
        results["batch_size"] = batch_size
        results["dropout_rate"] = dropout_rate

        for i in range(1, 3):
            step_start_time = time.time()
            tr_loss, tr_acc = model_utils.train(model, optim, dloader, loss_fn, device)
            log_utils.log_console(
                f"Epoch {i}, Node {user_id.num}, Train Loss: {tr_loss:.3f}, Train Acc: {tr_acc:.3f}"
            )

            # Send model to server
            model_bytes = serialize_model(model.state_dict())
            log_utils.log_console(f"Model bytes length: {len(model_bytes)}")
            log_utils.log_console(f"Using CUDA device: {torch.cuda.current_device()}")

            stub.SendMessage(
                comm_pb2.Message(
                    model=comm_pb2.Model(buffer=model_bytes),
                    id=user_id.id,
                )
            )
            log_utils.log_console("Message sent successfully.")

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

        # Log final results
        with open(os.path.join(output_dir, "final_results.txt"), "w") as f:
            for key, values in results.items():
                f.write(f"{key}: {values}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--model_arch", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--client_id", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--dropout_rate", type=float, required=True)
    parser.add_argument("--hparams_seed", type=int, required=True)
    parser.add_argument("--trial_seed", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--algorithm", type=str, default="fedavg")
    parser.add_argument("--gpu_index", type=int, required=True)
    args = parser.parse_args()
    run_client(args)
