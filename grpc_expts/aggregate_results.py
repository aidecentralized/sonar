import os
import json
import numpy as np
from collections import defaultdict
from tabulate import tabulate

BASE_LOG_DIR = "./sweep_logs"


def load_results(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading {file_path}: {e}")
        return None


def process_user_directory(user_path, aggregated_results, hparam_dir):
    results_path = os.path.join(user_path, "results.json")
    if os.path.exists(results_path):
        results = load_results(results_path)
        if results and "test_acc" in results and "train_indices" in results:
            test_acc_list = results["test_acc"]
            num_samples = len(results["train_indices"])
            model_name = results.get("model_arch", "unknown")
            dataset = results.get("dataset", "unknown")
            algo = results.get("algorithm", "unknown")
            for acc in test_acc_list:
                aggregated_results[hparam_dir]["test_acc"].append(acc)
                aggregated_results[hparam_dir]["num_samples"].append(num_samples)
                aggregated_results[hparam_dir]["model_name"] = model_name
                aggregated_results[hparam_dir]["dataset"] = dataset
                aggregated_results[hparam_dir]["algorithm"] = algo
        else:
            print(f"Missing 'test_acc' or 'train_indices' in {results_path}")


def process_trial_directory(trial_path, aggregated_results, hparam_dir):
    for user_dir in os.listdir(trial_path):
        user_path = os.path.join(trial_path, user_dir)
        if os.path.isdir(user_path):
            process_user_directory(user_path, aggregated_results, hparam_dir)


def aggregate_results(base_log_dir):
    aggregated_results = defaultdict(lambda: {"test_acc": [], "num_samples": [], "clients": set()})
    num_hparam_sets = 0
    num_trials = 0
    total_clients = set()

    for hparam_dir in os.listdir(base_log_dir):
        hparam_path = os.path.join(base_log_dir, hparam_dir)
        if os.path.isdir(hparam_path):
            num_hparam_sets += 1
            print(f"Processing hyperparameter directory: {hparam_dir}")
            if num_trials == 0:
                num_trials = len(os.listdir(hparam_path)) 
            for trial_dir in os.listdir(hparam_path):
                trial_path = os.path.join(hparam_path, trial_dir)
                if os.path.isdir(trial_path):
                    print(f"  Processing trial directory: {trial_dir}")
                    for user_dir in os.listdir(trial_path):
                        user_path = os.path.join(trial_path, user_dir)
                        if os.path.isdir(user_path):
                            process_user_directory(user_path, aggregated_results, hparam_dir)
                            total_clients.add(user_dir)

    return aggregated_results, num_hparam_sets, num_trials, len(total_clients)


def compute_statistics(aggregated_results):
    stats_results = {}

    for hparam_dir, data in aggregated_results.items():
        if data["test_acc"] and data["num_samples"]:
            test_acc_values = np.array(data["test_acc"])
            num_samples_values = np.array(data["num_samples"])

            mean_test_acc = np.mean(test_acc_values)
            std_test_acc = np.std(test_acc_values)

            total_samples = np.sum(num_samples_values)
            if total_samples > 0:
                weighted_mean_test_acc = (
                    np.sum(test_acc_values * num_samples_values) / total_samples
                )
                weighted_test_acc_values = (
                    test_acc_values * num_samples_values / total_samples
                )
                weighted_std_test_acc = np.std(weighted_test_acc_values)
            else:
                weighted_mean_test_acc = float("nan")
                weighted_std_test_acc = float("nan")

            stats_results[hparam_dir] = {
                "mean_test_accuracy": {"mean": mean_test_acc, "std": std_test_acc},
                "weighted_test_accuracy": {
                    "mean": weighted_mean_test_acc,
                    "std": weighted_std_test_acc,
                },
            }
        else:
            print(f"No valid data for hyperparameter directory: {hparam_dir}")

    return stats_results


def save_aggregated_results(stats_results, output_dir):
    print(stats_results)
    os.makedirs(output_dir, exist_ok=True)
    for hparam_dir, stats in stats_results.items():
        output_path = os.path.join(output_dir, f"{hparam_dir}_aggregated.json")
        try:
            with open(output_path, "w") as f:
                json.dump(stats, f, indent=4)
        except IOError as e:
            print(f"Error writing to {output_path}: {e}")


def find_best_hyperparameter_setting(stats_results):
    best_hparam_dir = None
    best_mean_accuracy = -np.inf

    for hparam_dir, stats in stats_results.items():
        mean_test_acc = stats["mean_test_accuracy"]["mean"]
        if mean_test_acc > best_mean_accuracy:
            best_mean_accuracy = mean_test_acc
            best_hparam_dir = hparam_dir

    return best_hparam_dir, best_mean_accuracy


if __name__ == "__main__":
    aggregated_results, num_hparam_sets, total_trials, num_clients = aggregate_results(BASE_LOG_DIR)
    stats_results = compute_statistics(aggregated_results)
    save_aggregated_results(stats_results, BASE_LOG_DIR)
    best_hparam_dir, best_mean_accuracy = find_best_hyperparameter_setting(stats_results)

    if best_hparam_dir:
        best_result = aggregated_results[best_hparam_dir]
        best_stats = stats_results[best_hparam_dir]["mean_test_accuracy"]
        model_name = best_result.get('model_name', 'unknown')
        dataset = best_result.get('dataset', 'unknown')
        algorithm = best_result.get('algorithm', 'unknown')

        print(f"\nBest hyperparameter setting: {best_hparam_dir} with mean test accuracy: {best_mean_accuracy:.4f}")
        print(f"Model: {model_name}")
        print(f"Dataset: {dataset}")
        print(f"Algorithm: {algorithm}")
        print(f"Best Test Accuracy: {best_mean_accuracy:.4f} ± {best_stats['std']:.4f}\n")

        # Prepare data for the table
        table_data = [[model_name, dataset, algorithm, f"{best_stats['mean']:.4f} ± {best_stats['std']:.4f}"]]

        # Define headers
        headers = ["Model", "Dataset", "Algorithm", "Best Test Accuracy"]

        # Print the table using tabulate
        print("\nTable:\n")
        print(tabulate(table_data, headers=headers, tablefmt="github"))

        # Print job statistics
        print("\nJob Statistics:\n")
        print(f"Number of hyperparameter sets: {num_hparam_sets}")
        print(f"Number of trials: {total_trials}")
        print(f"Number of clients: {num_clients}")