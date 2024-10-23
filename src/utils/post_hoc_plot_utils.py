import os
import pandas as pd
import numpy as np
from sklearn.metrics import auc
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt

# Define the path where your experiment logs are saved
log_path = '/u/jyuan24/sonar/src/expt_dump/cifar10_3users_333_0_malicious_test_gaussian_noise_seed1/logs/'

# Load Logs
def load_logs(node_id: str, metric_type: str) -> pd.DataFrame:
    """Loads the csv logs for a given node and metric (train/test, acc/loss)"""
    file_path = os.path.join(log_path, f'node_{node_id}/csv/{metric_type}.csv')
    return pd.read_csv(file_path)

def get_all_nodes() -> List[str]:
    """Return all node directories in the log folder"""
    return [d for d in os.listdir(log_path) if (os.path.isdir(os.path.join(log_path, d)) and d != "node_0" and d.startswith('node'))]

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

def compute_per_user_metrics(node_id: str) -> Dict[str, float]:
    """Computes AUC, best accuracy, and best loss for train/test."""
    train_acc = load_logs(node_id, 'train_acc')
    test_acc = load_logs(node_id, 'test_acc')
    train_loss = load_logs(node_id, 'train_loss')
    test_loss = load_logs(node_id, 'test_loss')

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

def aggregate_metrics_across_users(output_dir: str = f'{log_path}/aggregated_metrics/') -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Aggregate metrics across all users and save the results to CSV files."""
    nodes = get_all_nodes()
    all_metrics: List[Dict[str, float]] = []

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for node in nodes:
        node_id = node.split('_')[-1]
        metrics = compute_per_user_metrics(node_id)
        metrics['node'] = node
        all_metrics.append(metrics)

    # Convert to DataFrame for easier processing
    df_metrics = pd.DataFrame(all_metrics)
    
    # Calculate average and standard deviation
    avg_metrics = df_metrics.mean()
    std_metrics = df_metrics.std()

    # Save the DataFrame with per-user metrics
    df_metrics.to_csv(os.path.join(output_dir, 'per_user_metrics.csv'), index=False)

    # Save the average and standard deviation statistics
    summary_stats = pd.DataFrame({'Average': avg_metrics, 'Standard Deviation': std_metrics})
    summary_stats.to_csv(os.path.join(output_dir, 'summary_statistics.csv'))

    return avg_metrics, std_metrics, df_metrics

def compute_per_user_round_data(node_id: str, metrics_map: Optional[Dict[str, Dict[str, str]]] = None) -> Dict[str, np.ndarray]:
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
        data = load_logs(node_id, file_name)
        per_round_data[key] = data[file_name].values
        if 'rounds' not in per_round_data:
            per_round_data['rounds'] = data['iteration'].values

    return per_round_data

# Per Round Aggregation
def aggregate_per_round_data(metrics_map: Optional[Dict[str, Dict[str, str]]] = None) -> Dict[str, pd.DataFrame]:
    """Aggregate the per-round data for all users."""
    if metrics_map is None:
        metrics_map = {
            'train_acc': 'train_acc',
            'test_acc': 'test_acc',
            'train_loss': 'train_loss',
            'test_loss': 'test_loss',
        }

    nodes = get_all_nodes()
    all_users_data: Dict[str, List[np.ndarray]] = {metric: [] for metric in metrics_map}
    all_users_data['rounds'] = []

    for node in nodes:
        node_id = node.split('_')[-1]
        user_data = compute_per_user_round_data(node_id, metrics_map)

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

    # Plot the mean with standard deviation as a shaded area
    plt.plot(rounds, mean_metric, label='Average', color='black', linestyle='--')
    plt.fill_between(rounds, mean_metric - std_metric, mean_metric + std_metric, color='gray', alpha=0.2, label='Std dev')

    plt.xlabel('Rounds (Iterations)')
    plt.ylabel(ylabel)
    plt.title(f'{metric_name} per User and Aggregate')
    plt.legend()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f'{output_dir}{metric_name}_per_round.png')
    plt.close()

def plot_all_metrics(metrics_map: Optional[Dict[str, str]] = None) -> None:
    """Generates plots for all metrics over rounds with aggregation."""
    if metrics_map is None:
        metrics_map = {
            'test_acc': 'Test Accuracy',
            'train_acc': 'Train Accuracy',
            'test_loss': 'Test Loss',
            'train_loss': 'Train Loss'
        }

    all_users_data = aggregate_per_round_data()

    for key, display_name in metrics_map.items():
        plot_metric_per_round(all_users_data[key], all_users_data['rounds'], display_name, display_name.split()[1], f'{log_path}plots/')

    print("Plots saved as PNG files.")

if __name__ == "__main__":
    avg_metrics, std_metrics, df_metrics = aggregate_metrics_across_users()
    plot_all_metrics()