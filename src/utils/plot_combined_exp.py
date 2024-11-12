import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List

def combine_and_plot(
    exp_name: str, 
    experiment_map: Dict[str, str], 
    metrics_list: List[str], 
    plot_titles: List[str], 
    xlabels: List[str], 
    ylabels: List[str], 
    output_dir: str, 
    include_logs: bool = True
) -> None:
    """
    Combine and plot metrics from multiple experiments with 95% confidence intervals.
    
    Args:
        exp_name (str): Name of the experiment to be used in the output file names
        experiment_map (Dict[str, str]): Dictionary mapping experiment key names to file paths.
        metrics_list (List[str]): List of metrics to plot (e.g., ['train_acc', 'test_loss']).
        plot_titles (List[str]): List of titles for each plot in the same order as metrics_list.
        xlabels (List[str]): List of x-axis labels in the same order as metrics_list.
        ylabels (List[str]): List of y-axis labels in the same order as metrics_list.
        output_dir (str): Directory to save the plots and resulting logs.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(os.path.join(output_dir, 'plots')):
        os.makedirs(os.path.join(output_dir, 'plots'))
    
    for metric_index, metric_name in enumerate(metrics_list):
        plt.figure(figsize=(12, 8), dpi=300)
        
        for experiment_key, experiment_path in experiment_map.items():
            # Load the aggregated metric DataFrame for the experiment
            metric_df = pd.read_csv(os.path.join(experiment_path, f"{metric_name}_avg.csv"))
            # TODO: modify this to be CI
            # std_df = pd.read_csv(os.path.join(experiment_path, f"{metric_name}_std.csv"))
            ci_df = pd.read_csv(os.path.join(experiment_path, f"{metric_name}_ci95.csv"))

            # Assuming rounds or time steps are the index
            rounds = np.arange(len(metric_df))

            # Plot each experiment's mean with 95% CI
            mean_metric = metric_df.values.flatten()
            # std_metric = std_df.values.flatten()
            ci_95 = ci_df.values.flatten()

            # Plot the mean with confidence interval as a shaded area
            plt.plot(rounds, mean_metric, label=f'{experiment_key}', linestyle='--', linewidth=1.5)
            plt.fill_between(rounds, mean_metric - ci_95, mean_metric + ci_95, alpha=0.2)
            # plt.fill_between(rounds, mean_metric - std_metric, mean_metric + std_metric, alpha=0.2)

        # Plot customization
        plt.xlabel(xlabels[metric_index], fontsize=14)
        plt.ylabel(ylabels[metric_index], fontsize=14)
        plt.title(plot_titles[metric_index], fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)

        # Save the combined plot
        plt.savefig(os.path.join(output_dir, f"{exp_name}_{metric_name}_combined_plot.png"), bbox_inches='tight')
        plt.close()

        print(f"Combined plot for {exp_name} {metric_name} saved to {output_dir}")

        # Optionally, save the combined data for further analysis
        # combined_mean_df = pd.DataFrame({exp: pd.read_csv(os.path.join(path, f"{metric_name}_avg.csv")).values.flatten()
        #                                  for exp, path in experiment_map.items()})
        # combined_mean_df.to_csv(os.path.join(output_dir, f"{metric_name}_combined_avg.csv"), index=False)

        # combined_std_df = pd.DataFrame({exp: pd.read_csv(os.path.join(path, f"{metric_name}_std.csv")).values.flatten()
        #                                  for exp, path in experiment_map.items()})
        # combined_std_df.to_csv(os.path.join(output_dir, f"{metric_name}_combined_std.csv"), index=False)

        # combined_ci_df = pd.DataFrame({exp: pd.read_csv(os.path.join(path, f"{metric_name}_ci95.csv")).values.flatten()
        #                                for exp, path in experiment_map.items()})
        # combined_ci_df.to_csv(os.path.join(output_dir, f"{metric_name}_combined_ci95.csv"), index=False)

# Example usage for malicious attacks
if __name__ == "__main__":
    # /mas/camera/Experiments/SONAR/jyuan/4_attack_scaling/cifar10_40users_1250_bad_weights_0_malicious_seed1/logs/plots/test_acc_avg.csv

    base_dir = "/mas/camera/Experiments/SONAR/jyuan/4_attack_scaling"
    exp_names = ["bad_weights", "data_poisoning", "gradient_attack", "label_flip"]
    plot_names = ["Bad Weights Attack", "Data Poisoning Attack", "Gradient Attack", "Label Flip Attack"]
    num_malicious = [0, 1, 4, 8, 12]
    output_dir = "/mas/camera/Experiments/SONAR/jyuan/4_attack_scaling/plots/"

    for exp_ind, exp_name in enumerate(exp_names):
        experiment_map = {}
        for num_mal in num_malicious:
            experiment_map[f"{num_mal}_malicious"] = os.path.join(base_dir, f"cifar10_40users_1250_{exp_name}_{num_mal}_malicious_seed1/logs/plots/")
        metrics_list = ["test_acc"]
        plot_titles = [f"{plot_names[exp_ind]}: Test Accuracy Over Time"]
        xlabels = ["Rounds"]
        ylabels = ["Accuracy"]
        combine_and_plot(exp_name, experiment_map, metrics_list, plot_titles, xlabels, ylabels, output_dir)