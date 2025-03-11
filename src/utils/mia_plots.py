import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def collect_experiment_data(base_dir, mia_metric):
    """
    Collect and organize experiment data by both topology and alpha.
    Returns two dictionaries:
    1. Data organized by alpha -> topology
    2. Data organized by topology -> alpha
    """
    # Initialize dictionaries to store data
    data_by_alpha = {}  # For task 1: graphs per alpha with different topologies
    data_by_topology = {}  # For task 2: graphs per topology with different alphas
    
    # Process specific alpha directories
    alpha_dirs = ["0.1", "1.0", "100"]
    
    for alpha in alpha_dirs:
        alpha_path = os.path.join(base_dir, alpha)
        if not os.path.isdir(alpha_path):
            print(f"Alpha directory not found: {alpha_path}")
            continue
            
        print(f"Processing alpha directory: {alpha_path}")
            
        # Initialize alpha entry in the data_by_alpha dictionary
        data_by_alpha[alpha] = {}
        
        # Process all experiment directories in the alpha directory
        for exp in sorted(os.listdir(alpha_path)):
            exp_path = os.path.join(alpha_path, exp)
            
            # Skip if not a directory
            if not os.path.isdir(exp_path):
                continue
            
            # Extract topology from experiment name
            topology = exp.split("_")[0]
            
            print(f"  Processing experiment: {exp} (Topology: {topology})")
            
            # Initialize topology in data_by_topology if it doesn't exist
            if topology not in data_by_topology:
                data_by_topology[topology] = {}
                
            # Initialize topology in data_by_alpha[alpha] if it doesn't exist
            data_by_alpha[alpha][topology] = []
            
            # Initialize alpha in data_by_topology[topology] if it doesn't exist
            data_by_topology[topology][alpha] = []
            
            # Correct path to logs directory
            log_path = os.path.join(exp_path, "logs")
            
            # Skip if logs directory doesn't exist
            if not os.path.exists(log_path):
                print(f"    Logs directory not found: {log_path}")
                continue
            else:
                print(f"    Found logs directory: {log_path}")
                
            # Process each client directory
            client_metrics = []
            for client in sorted(os.listdir(log_path)):
                client_dir = os.path.join(log_path, client)
                
                # Skip if not a directory
                if not os.path.isdir(client_dir):
                    continue
                    
                # Path to metrics file
                metrics_file = os.path.join(client_dir, f"{mia_metric}_mia_stats_summary.json")
                
                # Skip if metrics file doesn't exist
                if not os.path.exists(metrics_file):
                    print(f"      Metrics file not found: {metrics_file}")
                    continue
                    
                # Load metrics
                try:
                    with open(metrics_file, "r") as f:
                        client_data = json.load(f)
                        client_metrics.append(client_data)
                        print(f"      Successfully loaded metrics from: {metrics_file}")
                except Exception as e:
                    print(f"      Error loading metrics from {metrics_file}: {e}")
                    continue
            
            # Skip if no client metrics were loaded
            if not client_metrics:
                print(f"    No client metrics found for {topology} in {alpha}")
                continue
                
            print(f"    Successfully processed {len(client_metrics)} clients for {topology} in {alpha}")
                
            # Process client metrics and store in both dictionaries
            data_by_alpha[alpha][topology] = client_metrics
            data_by_topology[topology][alpha] = client_metrics
    
    # Print summary of collected data
    print("\nData collection summary:")
    print("Alphas found:", list(data_by_alpha.keys()))
    print("Topologies found:", list(data_by_topology.keys()))
    
    for alpha, topologies in data_by_alpha.items():
        print(f"Alpha {alpha} has data for these topologies: {list(topologies.keys())}")
    
    for topology, alphas in data_by_topology.items():
        print(f"Topology {topology} has data for these alphas: {list(alphas.keys())}")
    
    return data_by_alpha, data_by_topology

def process_metrics(client_metrics):
    """
    Process client metrics to get median, min, and max values for each epoch.
    Returns epoch numbers, median values, min values, and max values.
    """
    # Convert string epoch keys to integers and sort them
    all_epochs = set()
    for client_data in client_metrics:
        all_epochs.update(int(epoch) for epoch in client_data.keys())
    
    epochs = sorted(all_epochs)
    
    # Initialize arrays to store metrics for each epoch
    median_values = []
    min_values = []
    max_values = []
    
    # Process each epoch
    for epoch in epochs:
        epoch_str = str(epoch)
        # Collect values for this epoch across all clients
        epoch_values = []
        
        for client_data in client_metrics:
            if epoch_str in client_data:
                epoch_values.append(client_data[epoch_str])
        
        if epoch_values:
            # Calculate median, min, and max
            median_values.append(np.median(epoch_values))
            min_values.append(np.min(epoch_values))
            max_values.append(np.max(epoch_values))
        else:
            # No data for this epoch, use NaN
            median_values.append(np.nan)
            min_values.append(np.nan)
            max_values.append(np.nan)
    
    return epochs, median_values, min_values, max_values

def create_alpha_graphs(data_by_alpha, mia_metric, output_dir="plots"):
    """
    Create one graph per alpha with all topologies overlaid (Task 1).
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the figure for all alphas
    fig, axes = plt.subplots(1, len(data_by_alpha), figsize=(6*len(data_by_alpha), 5), sharey=True)
    
    # If there's only one alpha, make axes iterable
    if len(data_by_alpha) == 1:
        axes = [axes]
    
    # Color palette for topologies
    colors = sns.color_palette("husl", len(set(topology for alpha_data in data_by_alpha.values() for topology in alpha_data)))
    topology_colors = {}
    
    # Get unique topologies across all alphas
    all_topologies = sorted(set(topology for alpha_data in data_by_alpha.values() for topology in alpha_data))
    for i, topology in enumerate(all_topologies):
        topology_colors[topology] = colors[i]
    
    print(f"\nCreating alpha graphs with these topologies: {all_topologies}")
    
    # Create one graph per alpha
    for i, (alpha, topology_data) in enumerate(sorted(data_by_alpha.items())):
        ax = axes[i]
        
        # Set title and labels
        ax.set_title(f"Alpha = {alpha}")
        ax.set_xlabel("Epoch")
        if i == 0:
            ax.set_ylabel(f"{mia_metric.replace('_', ' ').title()} ROC AUC")
        
        print(f"  Processing alpha={alpha} with topologies: {list(topology_data.keys())}")
        
        # Plot each topology
        for topology, client_metrics in sorted(topology_data.items()):
            if not client_metrics:
                print(f"    No data for topology={topology}")
                continue
                
            epochs, median_values, min_values, max_values = process_metrics(client_metrics)
            print(f"    Processed {len(epochs)} epochs for topology={topology}")
            
            # Plot median line
            line = ax.plot(epochs, median_values, label=topology, color=topology_colors[topology])
            
            # Plot min-max region
            ax.fill_between(epochs, min_values, max_values, alpha=0.2, color=line[0].get_color())
        
        # Add legend
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{mia_metric}_by_alpha.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved alpha graph to: {output_path}")
    plt.close()

def create_topology_graphs(data_by_topology, mia_metric, output_dir="plots"):
    """
    Create one graph per topology with all alphas overlaid (Task 2).
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the figure for all topologies
    fig, axes = plt.subplots(1, len(data_by_topology), figsize=(6*len(data_by_topology), 5), sharey=True)
    
    # If there's only one topology, make axes iterable
    if len(data_by_topology) == 1:
        axes = [axes]
    
    # Color palette for alphas
    colors = sns.color_palette("viridis", len(set(alpha for topology_data in data_by_topology.values() for alpha in topology_data)))
    alpha_colors = {}
    
    # Get unique alphas across all topologies
    all_alphas = sorted(set(alpha for topology_data in data_by_topology.values() for alpha in topology_data))
    for i, alpha in enumerate(all_alphas):
        alpha_colors[alpha] = colors[i]
    
    print(f"\nCreating topology graphs with these alphas: {all_alphas}")
    
    # Create one graph per topology
    for i, (topology, alpha_data) in enumerate(sorted(data_by_topology.items())):
        ax = axes[i]
        
        # Set title and labels
        ax.set_title(f"Topology = {topology}")
        ax.set_xlabel("Epoch")
        if i == 0:
            ax.set_ylabel(f"{mia_metric.replace('_', ' ').title()} ROC AUC")
        
        print(f"  Processing topology={topology} with alphas: {list(alpha_data.keys())}")
        
        # Plot each alpha
        for alpha, client_metrics in sorted(alpha_data.items()):
            if not client_metrics:
                print(f"    No data for alpha={alpha}")
                continue
                
            epochs, median_values, min_values, max_values = process_metrics(client_metrics)
            print(f"    Processed {len(epochs)} epochs for alpha={alpha}")
            
            # Plot median line
            line = ax.plot(epochs, median_values, label=f"Alpha={alpha}", color=alpha_colors[alpha])
            
            # Plot min-max region
            ax.fill_between(epochs, min_values, max_values, alpha=0.2, color=line[0].get_color())
        
        # Add legend
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{mia_metric}_by_topology.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved topology graph to: {output_path}")
    plt.close()

def main():
    # Configuration
    base_dir = "../expt_dump/mia/SGD/"
    mia_metric = "correct_prob"  # Options: entropy, loss, correct_prob
    output_dir = "plots"
    
    # Collect data
    print(f"\nCollecting experiment data from {base_dir}...")
    data_by_alpha, data_by_topology = collect_experiment_data(base_dir, mia_metric)
    
    # Create plots
    print(f"\nCreating graphs by alpha...")
    create_alpha_graphs(data_by_alpha, mia_metric, output_dir)
    
    print(f"\nCreating graphs by topology...")
    create_topology_graphs(data_by_topology, mia_metric, output_dir)
    
    print(f"\nAll plots saved to {output_dir}/")

if __name__ == "__main__":
    main()