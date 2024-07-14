import os
import json
import argparse
from collections import defaultdict


def find_client_dirs(output_dir):
    client_dirs = []
    print(f"Looking for logs directory in {output_dir}")
    for root, dirs, files in os.walk(output_dir):
        for dir_name in dirs:
            if dir_name.startswith("user_"):
                metrics_file = os.path.join(root, dir_name, "metrics.json")
                if os.path.exists(metrics_file):
                    print(f"Found metrics file: {metrics_file}")
                    client_dirs.append(metrics_file)
                else:
                    print(f"Metrics file not found in {os.path.join(root, dir_name)}")
    return client_dirs


def aggregate_metrics(client_dirs):
    aggregated_metrics = defaultdict(list)
    client_info = []
    for client_dir in client_dirs:
        with open(client_dir, "r") as f:
            client_metrics = json.load(f)
            client_info.append(
                {
                    "client_id": client_metrics["client_id"],
                    "gpu_index": client_metrics["gpu_index"],
                    "node_id": client_metrics["node_id"],
                    "train_indices": client_metrics["train_indices"],
                    "percentage_samples": client_metrics["percentage_samples"],
                    "trial_seed": client_metrics["trial_seed"],
                }
            )
            for key, values in client_metrics.items():
                if key not in [
                    "client_id",
                    "gpu_index",
                    "node_id",
                    "train_indices",
                    "percentage_samples",
                    "trial_seed",
                ]:
                    if isinstance(values, list):
                        aggregated_metrics[(key, client_metrics["trial_seed"])].extend(
                            values
                        )
                    else:
                        aggregated_metrics[(key, client_metrics["trial_seed"])].append(
                            values
                        )
    return aggregated_metrics, client_info


def compute_aggregated_results(aggregated_metrics, num_clients, num_trials):
    epoch_wise_test_acc = defaultdict(list)
    best_test_acc = 0
    trial_wise_test_acc = defaultdict(list)

    for (key, trial_seed), values in aggregated_metrics.items():
        if key == "test_acc":
            num_epochs = len(values) // num_clients
            for epoch in range(num_epochs):
                epoch_accs = values[epoch::num_epochs]
                epoch_avg_acc = sum(epoch_accs) / len(epoch_accs)
                epoch_wise_test_acc[epoch].append(epoch_avg_acc)
                trial_wise_test_acc[trial_seed].append(epoch_avg_acc)
                if epoch_avg_acc > best_test_acc:
                    best_test_acc = epoch_avg_acc

    avg_epoch_wise_test_acc = {
        epoch: sum(accs) / num_trials for epoch, accs in epoch_wise_test_acc.items()
    }
    avg_trial_wise_test_acc = {
        trial: sum(accs) / num_trials for trial, accs in trial_wise_test_acc.items()
    }

    aggregated_results = {
        "avg_epoch_wise_test_acc": avg_epoch_wise_test_acc,
        "avg_trial_wise_test_acc": avg_trial_wise_test_acc,
        "best_test_acc": best_test_acc,
    }

    return aggregated_results


def save_aggregated_results(aggregated_results, client_info, output_dir):
    results_path = os.path.join(output_dir, "aggregated_results.json")
    with open(results_path, "w") as f:
        json.dump(
            {"aggregated_metrics": aggregated_results, "client_info": client_info},
            f,
            indent=4,
        )
    print(f"Aggregated results saved to {results_path}")


def main(args):
    client_dirs = find_client_dirs(args.output_dir)
    if not client_dirs:
        print("No client metrics files found.")
        return
    aggregated_metrics, client_info = aggregate_metrics(client_dirs)
    num_clients = len(client_info) // len(
        set(info["trial_seed"] for info in client_info)
    )
    num_trials = len(set(info["trial_seed"] for info in client_info))
    aggregated_results = compute_aggregated_results(
        aggregated_metrics, num_clients, num_trials
    )
    save_aggregated_results(aggregated_results, client_info, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate metrics from multiple clients."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory containing the client output directories.",
    )
    args = parser.parse_args()
    main(args)
