import os
import pandas as pd
import numpy as np
from sklearn.metrics import auc
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import json
import networkx as nx
import imageio
from glob import glob

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
def plot_metric_per_round(metric_df: pd.DataFrame, rounds: np.ndarray, metric_name: str, ylabel: str, output_dir: str, plot_avg_only: bool = False) -> None:
    """Plot per-round data for each user and aggregate (mean and std)."""
    plt.figure(figsize=(12, 8), dpi=300)
    
    # Plot per-user data if plot_avg_only is False
    if not plot_avg_only:
        for col in metric_df.columns:
            plt.plot(rounds, metric_df[col], alpha=0.6)

    # Compute mean and std and 95% confidence interval
    mean_metric = metric_df.mean(axis=1)
    std_metric = metric_df.std(axis=1)
    n = metric_df.shape[1]
    ci_95 = 1.96 * (std_metric / np.sqrt(n))

    # Save the mean and std
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    mean_metric.to_csv(f'{output_dir}{metric_name}_avg.csv', index=False)
    std_metric.to_csv(f'{output_dir}{metric_name}_std.csv', index=False)
    ci_95.to_csv(f'{output_dir}{metric_name}_ci95.csv', index=False)


    # Plot the mean with standard deviation as a shaded area
    plt.plot(rounds, mean_metric, label='Average', color='black', linestyle='--')
    # plt.fill_between(rounds, mean_metric - std_metric, mean_metric + std_metric, color='gray', alpha=0.2, label='Std dev')
    plt.fill_between(rounds, mean_metric - ci_95, mean_metric + ci_95, color='gray', alpha=0.2, label='95% CI')

    # Set labels, title, and add grid for better readability
    plt.xlabel('Rounds (Iterations)', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(f'{ylabel} per User and Aggregate', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)

    # Save the plot
    plt.savefig(f'{output_dir}{metric_name}_per_round.png', bbox_inches='tight')
    plt.close()

def compute_per_user_realtime_data(
    node_id: str, 
    logs_dir: str, 
    time_interval: int, 
    num_ticks: Optional[int] = None
) -> Dict[str, pd.Series]:
    """
    Optimized computation of per-user real-time data based on elapsed time and logged metrics per round.
    
    Args:
        node_id (str): ID of the node (user).
        logs_dir (str): Directory path where logs are stored.
        time_interval (int): Interval in seconds for each tick in the real-time plot.
        num_ticks (Optional[int]): Total number of ticks to fill. If specified, fills remaining ticks with last known value.
    
    Returns:
        Dict[str, pd.Series]: A dictionary with real-time metrics Series for each metric, indexed by time.
    """
    # Load time elapsed data
    time_data = load_logs(node_id, 'time_elapsed', logs_dir)
    round_times = time_data['time_elapsed'].values
    rounds = time_data['iteration'].values

    # Compute per-round data for the metrics
    per_round_data = compute_per_user_round_data(node_id, logs_dir)

    # Initialize per_time_data for each metric
    per_time_data = {key: [] for key in per_round_data.keys() if key != 'iteration'}
    
    # Determine maximum time based on the final value in round_times and calculate the number of ticks
    max_time = round_times[-1] if len(round_times) > 0 else 0
    calculated_ticks = int(max_time // time_interval + 1)
    total_ticks = num_ticks if num_ticks is not None else calculated_ticks

    # Initialize a pointer for the current round index
    round_idx = 0
    time_ticks = [tick * time_interval for tick in range(1, total_ticks + 1)]

    # Loop through each time tick based on the time_interval
    for current_time in time_ticks:
        # Move round_idx forward until round_times[round_idx] > current_time
        while round_idx < len(round_times) and round_times[round_idx] <= current_time:
            round_idx += 1
        # Use the last valid round's metrics for the current tick
        latest_round_idx = round_idx - 1 if round_idx > 0 else None
        
        for key in per_time_data.keys():
            if latest_round_idx is not None:
                per_time_data[key].append(per_round_data[key][latest_round_idx])
            else:
                per_time_data[key].append(np.nan)  # Start with NaN if no valid data exists initially

    # Fill remaining ticks with the last known value for each metric
    for key in per_time_data.keys():
        if per_time_data[key]:  # Check if there’s any data collected
            last_value = per_time_data[key][-1]
            per_time_data[key].extend([last_value] * (total_ticks - len(per_time_data[key])))

    # Convert lists to Series with time_ticks as the index
    per_time_data = {
        key: pd.Series(data=values, index=time_ticks) for key, values in per_time_data.items()
    }
    
    return per_time_data


def aggregate_per_realtime_data(
    logs_dir: str, 
    metrics_map: Optional[Dict[str, str]] = None, 
    time_interval: Optional[int] = None, 
    num_ticks: Optional[int] = 200,
) -> Dict[str, pd.DataFrame]:
    """
    Aggregate the per-time data for all users.
    
    Args:
        logs_dir (str): Directory path where logs are stored.
        metrics_map (Optional[Dict[str, str]]): Mapping of metric names to file names.
        time_interval (Optional[int]): Interval in seconds for each tick in the real-time plot.
        num_ticks (Optional[int]): Number of ticks to display. Used if time_interval is not provided.
    
    Returns:
        Dict[str, pd.DataFrame]: A dictionary with real-time metrics DataFrames for each metric, indexed by time.
    """
    if metrics_map is None:
        metrics_map = {
            'train_acc': 'train_acc',
            'test_acc': 'test_acc',
            'train_loss': 'train_loss',
            'test_loss': 'test_loss',
        }

    nodes = get_all_nodes(logs_dir)

    # Step 1: Determine max time_elapsed across all nodes if num_ticks is given and time_interval is None
    if time_interval is None:
        max_elapsed_time = 0
        for node in nodes:
            node_id = node.split('_')[-1]
            time_data = load_logs(node_id, 'time_elapsed', logs_dir)
            max_elapsed_time = max(max_elapsed_time, time_data['time_elapsed'].max())
        
        # Calculate time_interval based on max_elapsed_time and num_ticks
        time_interval = max_elapsed_time // num_ticks
    
    # Initialize aggregated data storage
    all_users_data = {metric: [] for metric in metrics_map}
    time_ticks = None

    # Step 2: Aggregate per-user data based on computed time_interval
    for node in nodes:
        node_id = node.split('_')[-1]
        user_data = compute_per_user_realtime_data(node_id, logs_dir, time_interval, num_ticks=num_ticks)
        
        # Append data from each user
        for key in metrics_map:
            all_users_data[key].append(user_data[key].values)  # Each should be of shape (num_ticks,)

        # Record time_ticks only once (they will be the same for all users)
        if time_ticks is None:
            time_ticks = user_data[list(metrics_map.keys())[0]].index.values

    # write all user data as a file to check
    with open('./all_users_data.csv', 'w') as f:
        pd.DataFrame(all_users_data).to_csv(f)

    # Convert lists of arrays into DataFrames for each metric
    aggregated_data = {
        key: pd.DataFrame(np.stack(all_users_data[key], axis=1), index=time_ticks)
        for key in metrics_map
    }

    return aggregated_data


def plot_metric_per_realtime(metric_df: pd.DataFrame, time_ticks: np.ndarray, metric_name: str, ylabel: str, output_dir: str, plot_avg_only: bool = False) -> None:
    """
    Plot per-time elapsed data for each user and aggregate (mean and std).
    
    Args:
        metric_df (pd.DataFrame): DataFrame containing the metric data for each user (one column per user).
        time_ticks (np.ndarray): Array of time elapsed values for each tick.
        metric_name (str): Name of the metric (e.g., 'train_acc', 'test_loss').
        ylabel (str): Label for the y-axis of the plot.
        output_dir (str): Directory to save the plot and CSV files.
    """
    plt.figure(figsize=(12, 8), dpi=300)
    
    # Plot per-user data
    if not plot_avg_only:
        for col in metric_df.columns:
            # plt.plot(time_ticks, metric_df[col], alpha=0.6, label=f'User {col+1}')
            plt.plot(time_ticks, metric_df[col], alpha=0.6)

    # Compute mean and std and 95% confidence interval
    mean_metric = metric_df.mean(axis=1)
    std_metric = metric_df.std(axis=1)
    n = metric_df.shape[1]
    ci_95 = 1.96 * (std_metric / np.sqrt(n))

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the mean and std to CSV
    mean_metric.to_csv(f'{output_dir}{metric_name}_avg_per_time.csv', index=False)
    std_metric.to_csv(f'{output_dir}{metric_name}_std_per_time.csv', index=False)
    ci_95.to_csv(f'{output_dir}{metric_name}_ci95.csv', index=False)

    # Plot the mean with standard deviation as a shaded area
    plt.plot(time_ticks, mean_metric, label='Average', color='black', linestyle='--')
    # plt.fill_between(time_ticks, mean_metric - std_metric, mean_metric + std_metric, color='gray', alpha=0.2, label='Std dev')
    plt.fill_between(time_ticks, mean_metric - ci_95, mean_metric + ci_95, color='gray', alpha=0.2, label='95% CI')

    # Set labels and title
    plt.xlabel('Time Elapsed (seconds)', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(f'{ylabel} per User and Aggregate over Time Elapsed', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    
    # Save the plot
    plt.savefig(f'{output_dir}{metric_name}_per_time.png')
    plt.close()



def create_weighted_images(neighbors, output_dir: str, pos):
    """
    Create the images for the network visualization.
    
    Parameters:
    - neighbors: 3d numpy array of neighbors for each node x - round, y - node, z - neighbors
    """
    #create a network x graph and visualize it for each round

    freq = np.zeros((neighbors.shape[1], neighbors.shape[1]))
    for round in range(neighbors.shape[0]):

        for node in range(neighbors.shape[1]):
            for neighbor in neighbors[round][node]:
                freq[node][neighbor-1] += 1
                
        # Create the directed graph 
        graph = nx.DiGraph()
        #add edges based on which edges in freq are greater than 0 and use that as the weight
        for i in range(neighbors.shape[1]):
            for j in range(neighbors.shape[1]):
                if freq[i][j] > 0:
                    graph.add_edge(i + 1, j + 1, weight=3 * freq[i][j])


            
        
        # draw nodes 
        nx.draw_networkx_nodes(graph, pos, node_size=700, node_color="skyblue")
        nx.draw_networkx_labels(graph, pos, font_size=8, font_weight="bold")

        #make opposite edges not overlap by adding curvature and make edges thicker based on frequency
        curvatureDict = {}
        for _, (u, v) in enumerate(graph.edges()):
            # make sure u v and v u always have different curvature
            if (u,v) not in curvatureDict:
                curvatureDict[(u,v)] = 0.1
                curvatureDict[(v,u)] = 0.1
            
            rad = curvatureDict[(u,v)]
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=[(u, v)],
                connectionstyle=f"arc3,rad={rad}",
                width=freq[u-1][v-1]/3,
                arrows=True,
                arrowsize=20
            )

        # Create the image
        plt.title(f"Round {round + 1}")
        plt.savefig(f"{output_dir}/weighted_graph_{round + 1}.png")
    
    plt.close()

def create_images(neighbors, output_dir: str, pos):
    """
    Create the images for the network visualization.
    
    Parameters:
    - neighbors: 3d numpy array of neighbors for each node x - round, y - node, z - neighbors
    """
    #create a network x graph and visualize it for each round
    for round in range(neighbors.shape[0]):

        # Create the directed graph 
        graph = nx.DiGraph()
        for node in range(neighbors.shape[1]):
            for neighbor in neighbors[round][node]:
                graph.add_edge(node + 1, neighbor)
        
        # draw nodes 
        nx.draw_networkx_nodes(graph, pos, node_size=700, node_color="skyblue")
        nx.draw_networkx_labels(graph, pos, font_size=8, font_weight="bold")


        #make opposite edges not overlap by adding curvature
        curvatureDict = {}
        for i, (u, v) in enumerate(graph.edges()):
            # make sure u v and v u always have different curvature
            if (u,v) not in curvatureDict:
                curvatureDict[(u,v)] = 0.1
                curvatureDict[(v,u)] = 0.1
            
            rad = curvatureDict[(u,v)]
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=[(u, v)],
                connectionstyle=f"arc3,rad={rad}",
                arrows=True,
                arrowsize=20
            )

        # Create the image
        plt.title(f"Round {round + 1}")
        plt.savefig(f"{output_dir}/graph_{round + 1}.png")
        plt.close()

def create_video(output_dir: str, image_name: str):
    """Create a gif from the images."""
    images = []
    for filename in sorted(glob(f"{output_dir}/{image_name}_*.png")):
        images.append(imageio.imread(filename))
    imageio.mimsave(f"{output_dir}/{image_name}_video.gif", images, fps = 1, loop = 0)



def create_heatmap(neighbors, output_dir: str):
        """
        Create a heatmap of the edge frequency.
        
        Parameters:
        - neighbors: 3d numpy array of neighbors for each node x - round, y - node, z - neighbors
        """

        # Initialize the edge frequency matrix
        edge_frequency_matrix = np.zeros((neighbors.shape[1]+1, neighbors.shape[1]+1))
        # Iterate over all the rounds
        for round in range(neighbors.shape[0]):
            # Iterate over all the nodes
            for node in range(neighbors.shape[1]):
                # Iterate over all the
                for neighbor in neighbors[round][node]:
                    edge_frequency_matrix[node+1][neighbor] += 1

        edge_frequency_matrix = np.log(edge_frequency_matrix + 1)  # Log scale for better visualization
        # Create the heatmap
        plt.figure(figsize=(10, 6))
        plt.imshow(edge_frequency_matrix, cmap="hot", interpolation="nearest")
        plt.title("Edge Frequency Matrix")
        plt.colorbar(label="Frequency of Communication")
        plt.xlabel("Node")
        plt.ylabel("Node")
        plt.xticks(range(1,neighbors.shape[1]+1))
        plt.yticks(range(1,neighbors.shape[1]+1))
        plt.savefig(f"{output_dir}/edge_frequency_heatmap.png")
        plt.close()

def plot_all_metrics(logs_dir: str, per_round: bool = True, per_time: bool = True, metrics_map: Optional[Dict[str, str]] = None, plot_avg_only: bool=False, **kwargs) -> None:
    """Generates plots for all metrics over rounds with aggregation."""
    if metrics_map is None:
        metrics_map = {
            'test_acc': 'Test Accuracy',
            'train_acc': 'Train Accuracy',
            'test_loss': 'Test Loss',
            'train_loss': 'Train Loss'
        }

    if per_round:
        all_users_data = aggregate_per_round_data(logs_dir, **kwargs)

        for key, display_name in metrics_map.items():
            plot_metric_per_round(
                metric_df=all_users_data[key], 
                rounds=all_users_data['rounds'], 
                metric_name=key, 
                ylabel=display_name, 
                output_dir=f'{logs_dir}/plots/',
                plot_avg_only=plot_avg_only,
                **kwargs
                )

    if per_time:
        time_data = os.path.join(logs_dir, 'node_1/csv/time_elapsed.csv')
        # check if time elapsed data exists
        if not os.path.exists(time_data):
            print("Time elapsed data not found. Skipping per-time plotting.")
            return
        all_users_data = aggregate_per_realtime_data(logs_dir, **kwargs)

        for key, display_name in metrics_map.items():
            plot_metric_per_realtime(
                metric_df=all_users_data[key], 
                time_ticks=all_users_data[key].index.values, 
                metric_name=key, 
                ylabel=display_name, 
                output_dir=f'{logs_dir}/plots/',
                plot_avg_only=plot_avg_only,
                **kwargs
                )
            
    neighbors = aggregate_neighbors_across_users(logs_dir)
    # create_heatmap(neighbors, f'{os.path.dirname(logs_dir)}/plots/')
    pos = nx.spring_layout(nx.DiGraph({i+1: [] for i in range(neighbors.shape[1])}))
    create_images(neighbors, f'{os.path.dirname(logs_dir)}/plots/', pos)
    create_weighted_images(neighbors, f'{os.path.dirname(logs_dir)}/plots/', pos)
    create_video(f'{os.path.dirname(logs_dir)}/plots/', 'graph')
    create_video(f'{os.path.dirname(logs_dir)}/plots/', 'weighted_graph')
    create_heatmap(neighbors, f'{os.path.dirname(logs_dir)}/plots/')
    

    print("Plots saved as PNG files.")

def aggregate_neighbors_across_users(logs_dir: str) -> np.ndarray:
    """Aggregate the neighbors of each node across all users."""
    nodes = get_all_nodes(logs_dir)
    nodes.sort()  # Sort the nodes to ensure consistent order

    all_users_neighbors = []

    for node in nodes:
        node_id = node.split('_')[-1]
        neighbors_file = os.path.join(logs_dir, f'node_{node_id}/csv/neighbors.csv')
        neighbors = pd.read_csv(neighbors_file)
        np.array(all_users_neighbors.append(neighbors['neighbors'].apply(json.loads).values))

    return np.array(all_users_neighbors).T

# Use if you a specific experiment folder
# if __name__ == "__main__":
#     # Define the path where your experiment logs are saved
#     logs_dir = '/mas/camera/Experiments/SONAR/jyuan/experiment/logs_sample_time_elapsed/'
#     avg_metrics, std_metrics, df_metrics = aggregate_metrics_across_users(logs_dir)
#     plot_all_metrics(logs_dir, per_round=True, per_time=True, plot_avg_only=True)


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
                plot_all_metrics(logs_dir, per_round=True, per_time=True, plot_avg_only=True)
            except Exception as e:
                print(f"Error processing {logs_dir}: {e}")
                continue