import os
import pandas as pd
import numpy as np
from sklearn.metrics import auc
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import json

# Load Logs
def load_logs(node_id: str, metric_type: str, logs_dir: str) -> pd.DataFrame:
    """Loads the csv logs for a given node and metric (train/test, acc/loss)"""
    file_path = os.path.join(logs_dir, f'node_{node_id}/csv/{metric_type}.csv')
    return pd.read_csv(file_path)

def get_all_nodes(logs_dir: str) -> List[str]:
    """Return all node directories in the log folder"""
    return [d for d in os.listdir(logs_dir) if (os.path.isdir(os.path.join(logs_dir, d)) and d != "node_0" and d.startswith('node'))]

def get_node_type(node_id: str, logs_dir: str) -> str:
    """Determine if the node is malicious or normal based on its config file."""
    config_path = os.path.join(logs_dir, f'node_{node_id}', 'config.json')
    with open(config_path, 'r') as file:
        config = json.load(file)
    
    # Check if node has a malicious type
    return config.get("malicious_type", "normal")

# Calculate Metrics Per User
def calculate_auc(df: pd.DataFrame, metric: str = 'acc') -> float:
    """Calculate AUC for the given dataframe's accuracy or loss."""
    return auc(df['iteration'], df[metric])

def best_accuracy(df: pd.DataFrame, metric: str = 'acc') -> float:
    """Find the best test accuracy or lowest loss for a given metric."""
    return df[metric].max()

def best_loss(df: pd.DataFrame, metric: str) -> float:
    """Find the lowest loss for a given metric."""
    return df[metric].min()

def compute_per_user_metrics(node_id: str, logs_dir: str) -> Dict[str, float]:
    """Computes AUC, best accuracy, and best loss for train/test."""
    train_acc = load_logs(node_id, 'train_acc', logs_dir)
    test_acc = load_logs(node_id, 'test_acc', logs_dir)
    train_loss = load_logs(node_id, 'train_loss', logs_dir)
    test_loss = load_logs(node_id, 'test_loss', logs_dir)

    metrics = {
        'train_auc_acc': calculate_auc(train_acc, 'train_acc'),
        'test_auc_acc': calculate_auc(test_acc, 'test_acc'),
        'train_auc_loss': calculate_auc(train_loss, 'train_loss'),
        'test_auc_loss': calculate_auc(test_loss, 'test_loss'),
        'best_train_acc': best_accuracy(train_acc, 'train_acc'),
        'best_test_acc': best_accuracy(test_acc, 'test_acc'),
        'best_train_loss': best_loss(train_loss, 'train_loss'),
        'best_test_loss': best_loss(test_loss, 'test_loss')
    }

    return metrics

def aggregate_metrics_across_users(logs_dir: str, output_dir: Optional[str] = None) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Aggregate metrics across all users, categorize nodes, and save the results to CSV files."""
    nodes = get_all_nodes(logs_dir)
    all_metrics: List[Dict[str, float]] = []
    normal_metrics = []
    malicious_metrics = []

    # Ensure the output directory exists
    if not output_dir:
        output_dir = os.path.join(logs_dir, 'aggregated_metrics')
    os.makedirs(output_dir, exist_ok=True)

    for node in nodes:
        node_id = node.split('_')[-1]
        metrics = compute_per_user_metrics(node_id, logs_dir)
        metrics['node'] = node
        
        # Get node type and categorize metrics
        node_type = get_node_type(node_id, logs_dir)
        if node_type == "normal":
            normal_metrics.append(metrics)
        elif node_type != "normal":  # Check for malicious nodes
            malicious_metrics.append(metrics)
        
        all_metrics.append(metrics)

    # Convert to DataFrame for easier processing
    df_metrics = pd.DataFrame(all_metrics)
    numeric_columns = df_metrics.select_dtypes(include=[np.number])

    # Calculate overall average and standard deviation
    overall_avg = numeric_columns.mean()
    overall_std = numeric_columns.std()

    # Convert normal and malicious metrics to DataFrames
    df_normal = pd.DataFrame(normal_metrics)
    df_malicious = pd.DataFrame(malicious_metrics)

    # Calculate normal node statistics
    normal_avg = df_normal.mean(numeric_only=True)
    normal_std = df_normal.std(numeric_only=True)

    # Calculate malicious node statistics (if there are malicious nodes)
    if not df_malicious.empty:
        malicious_avg = df_malicious.mean(numeric_only=True)
        malicious_std = df_malicious.std(numeric_only=True)
    else:
        # If no malicious nodes, fill with NaN values for clarity
        malicious_avg = pd.Series([np.nan] * len(normal_avg), index=normal_avg.index)
        malicious_std = pd.Series([np.nan] * len(normal_std), index=normal_std.index)

    # Compile all metrics into a DataFrame with descriptive row names
    summary_stats_data = {
        f"{metric}_overall": [overall_avg[metric], overall_std[metric]]
        for metric in overall_avg.index
    }
    summary_stats_data.update({
        f"{metric}_normal": [normal_avg[metric], normal_std[metric]]
        for metric in normal_avg.index
    })
    summary_stats_data.update({
        f"{metric}_malicious": [malicious_avg[metric], malicious_std[metric]]
        for metric in malicious_avg.index
    })

    # Create a DataFrame from the dictionary and specify columns
    summary_stats_df = pd.DataFrame.from_dict(summary_stats_data, orient='index', columns=["Average", "Standard Deviation"])

    # Save the summary statistics to a single CSV file
    summary_stats_df.to_csv(os.path.join(output_dir, 'summary_statistics.csv'))

    return overall_avg, overall_std, df_metrics

def compute_per_user_round_data(node_id: str, logs_dir: str, metrics_map: Optional[Dict[str, str]] = None) -> Dict[str, np.ndarray]:
    """Extract per-round data (accuracy and loss) for train/test from the logs."""
    if metrics_map is None:
        metrics_map = {
            'train_acc': 'train_acc',
            'test_acc': 'test_acc',
            'train_loss': 'train_loss',
            'test_loss': 'test_loss',
        }

    per_round_data = {}
    for key, file_name in metrics_map.items():
        data = load_logs(node_id, file_name, logs_dir)
        per_round_data[key] = data[file_name].values
        if 'rounds' not in per_round_data:
            per_round_data['rounds'] = data['iteration'].values

    return per_round_data

# Per Round Aggregation
def aggregate_per_round_data(logs_dir: str, metrics_map: Optional[Dict[str, str]] = None) -> Dict[str, pd.DataFrame]:
    """Aggregate the per-round data for all users."""
    if metrics_map is None:
        metrics_map = {
            'train_acc': 'train_acc',
            'test_acc': 'test_acc',
            'train_loss': 'train_loss',
            'test_loss': 'test_loss',
        }

    nodes = get_all_nodes(logs_dir)
    all_users_data: Dict[str, List[np.ndarray]] = {metric: [] for metric in metrics_map}
    all_users_data['rounds'] = []

    for node in nodes:
        node_id = node.split('_')[-1]
        user_data = compute_per_user_round_data(node_id, logs_dir, metrics_map)

        # Collect data for all users
        for key in metrics_map:
            all_users_data[key].append(user_data[key])

    # Convert lists of arrays into DataFrames for easier aggregation
    rounds = user_data['rounds']  # All users should have the same rounds
    all_users_data['rounds'] = rounds

    for key in metrics_map:
        all_users_data[key] = pd.DataFrame(all_users_data[key]).transpose()

    return all_users_data

# Plotting
def plot_metric_per_round(metric_df: pd.DataFrame, rounds: np.ndarray, metric_name: str, ylabel: str, output_dir: str) -> None:
    """Plot per-round data for each user and aggregate (mean and std)."""
    plt.figure(figsize=(10, 6))
    
    # Plot per-user data
    for col in metric_df.columns:
        plt.plot(rounds, metric_df[col], alpha=0.6, label=f'User {col+1}')

    # Compute mean and std
    mean_metric = metric_df.mean(axis=1)
    std_metric = metric_df.std(axis=1)

    # Save the mean and std
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    mean_metric.to_csv(f'{output_dir}{metric_name}_avg.csv', index=False)
    std_metric.to_csv(f'{output_dir}{metric_name}_std.csv', index=False)


    # Plot the mean with standard deviation as a shaded area
    plt.plot(rounds, mean_metric, label='Average', color='black', linestyle='--')
    plt.fill_between(rounds, mean_metric - std_metric, mean_metric + std_metric, color='gray', alpha=0.2, label='Std dev')

    plt.xlabel('Rounds (Iterations)')
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} per User and Aggregate')
    plt.legend()
    plt.savefig(f'{output_dir}{metric_name}_per_round.png')
    plt.close()

def plot_all_metrics(logs_dir: str, metrics_map: Optional[Dict[str, str]] = None) -> None:
    """Generates plots for all metrics over rounds with aggregation."""
    if metrics_map is None:
        metrics_map = {
            'test_acc': 'Test Accuracy',
            'train_acc': 'Train Accuracy',
            'test_loss': 'Test Loss',
            'train_loss': 'Train Loss'
        }

    all_users_data = aggregate_per_round_data(logs_dir)

    for key, display_name in metrics_map.items():
        plot_metric_per_round(
            metric_df=all_users_data[key], 
            rounds=all_users_data['rounds'], 
            metric_name=key, 
            ylabel=display_name, 
            output_dir=f'{logs_dir}plots/'
            )

    print("Plots saved as PNG files.")

# Use if you a specific experiment folder
# if __name__ == "__main__":
#     # Define the path where your experiment logs are saved
#     logs_dir = '/mas/camera/Experiments/SONAR/abhi/cifar10_36users_1250_convergence_ringm3_seed2/logs/'
#     avg_metrics, std_metrics, df_metrics = aggregate_metrics_across_users(logs_dir)
#     plot_all_metrics(logs_dir)


# Use if you want to compute for multiple experiment folders
if __name__ == "__main__":
    # Define the base directory where your experiment logs are saved
    base_logs_dir = '/mas/camera/Experiments/SONAR/abhi/'

    # Iterate over each subdirectory in the base directory
    for experiment_folder in os.listdir(base_logs_dir):
        experiment_path = os.path.join(base_logs_dir, experiment_folder)
        logs_dir = os.path.join(experiment_path, 'logs')

        if os.path.isdir(logs_dir):
            try:
                print(f"Processing logs in: {logs_dir}")
                avg_metrics, std_metrics, df_metrics = aggregate_metrics_across_users(logs_dir)
                plot_all_metrics(logs_dir)
            except Exception as e:
                print(f"Error processing {logs_dir}: {e}")
                continue