import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1 import AxesGrid
import math

import logging

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


class PlotUtils:
    def __init__(self, config, with_title=True) -> None:
        self.plot_dir = config["plot_path"]
        if not os.path.exists(self.plot_dir) or not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        self.config = config
        self.with_title = with_title

    def get_dataset_config_string(self):
        if isinstance(self.config["dset"], dict):
            dsets = sorted(list(set(self.config["dset"].values())))
            nodes_by_dset = {dset: [] for dset in dsets}
            for node in self.config["dset"].keys():
                nodes_by_dset[self.config["dset"][node]].append(node)

    def plot_clients_stats_per_round(self, stats, round_step, metric):
        plt.figure()
        x_axis = np.array(range(stats.shape[1])) * round_step
        for client in range(stats.shape[0]):
            plt.plot(x_axis, stats[client], label=f"client{client+1}")
        plt.legend()
        plt.xlabel("Round")
        plt.ylabel(metric)
        if self.with_title:
            plt.title(
                f"{metric} - {self.config['algo']}-{self.config['dset_name']}-samples per clients: {self.config['samples_per_user']}\nlabel distribution: {self.config['train_label_distribution']}"
            )
        else:
            plt.title(f"{metric}")
        plt.savefig(f"{self.plot_dir}/{metric}_per_round.png")
        plt.close()

    def plot_clients_avg_stats_per_round(self, stats, round_step, metric):
        plt.figure()
        x_axis = np.array(range(stats.shape[1])) * round_step
        # Plot mean and quantiles
        plt.plot(x_axis, np.mean(stats, axis=0), label="mean")
        plt.fill_between(
            x_axis,
            np.quantile(stats, 0.25, axis=0),
            np.quantile(stats, 0.75, axis=0),
            alpha=0.3,
        )
        plt.xlabel("Round")
        plt.ylabel(metric)
        if self.with_title:
            plt.title(
                f"avg_{metric} - {self.config['algo']}-{self.config['dset_name']}-clients:{self.config['num_users']}-samples per clients: {self.config['samples_per_user']}\nlabel distribution: {self.config['train_label_distribution']}"
            )
        else:
            plt.title(f"avg_{metric}")
        plt.savefig(f"{self.plot_dir}/avg_{metric}_per_round.png")
        plt.close()

    def plot_avg_clients_weights_heatmap(
        self, weights_stats, name, x_label, diagonal=None
    ):
        # Weights array is indexed by [client, round, client]
        # We want to plot the mean of the weights across all rounds
        mean_weights = np.mean(weights_stats, axis=1)
        num_client = weights_stats.shape[0]

        if diagonal is not None:
            if diagonal == "zero":
                np.fill_diagonal(mean_weights, 0)
            else:
                raise ValueError(f"Unknown diagonal value {diagonal}")

        plt.figure()
        plt.imshow(mean_weights, cmap="hot", interpolation="nearest")
        plt.xticks(range(0, num_client), range(1, num_client + 1))
        plt.yticks(range(0, num_client), range(1, num_client + 1))
        plt.colorbar()
        plt.xlabel(x_label)
        plt.ylabel("Client")
        if self.with_title:
            plt.title(
                f"{name} weights - {self.config['algo']}-{self.config['dset_name']}-clients:{self.config['num_users']}-samples per clients: {self.config['samples_per_user']}\nlabel distribution: {self.config['train_label_distribution']}"
            )
        else:
            plt.title(f"{name} weights")

        plt.savefig(
            f"{self.plot_dir}/{str.lower(name).replace(' ', '_')}{('_'+diagonal) if diagonal is not None else ''}.png"
        )
        plt.close()

    def plot_clients_weights_heatmap_over_time(
        self,
        weights_stats,
        name,
        x_label,
        num_plot=10,
        log_space=False,
        single_round=True,
        plot_rounds=None,
        diagonal=None,
    ):
        # Plot multiple heatmaps in one figure, evenly spaced over time
        # Collab weights array is indexed by [client, round, client]

        num_round = weights_stats.shape[1]

        num_plot = min(num_plot, num_round)

        if plot_rounds is not None:
            num_plot = len(plot_rounds)
        elif log_space:
            plot_rounds = np.logspace(
                0, np.log10(num_round), num_plot + (0 if single_round else 1), dtype=int
            )
            plot_rounds = list(dict.fromkeys(plot_rounds))  # remove duplicates
        else:
            plot_rounds = np.linspace(
                0, num_round - 1, num_plot + (0 if single_round else 1), dtype=int
            )

        vmin = np.min(weights_stats)
        vmax = np.max(weights_stats)

        single_color_bar = np.max(weights_stats, axis=(0, 2)).mean() == vmax

        # Plot multiple heatmaps in one figure
        fig = plt.figure(figsize=(20, 10))

        grid = AxesGrid(
            fig,
            111,
            nrows_ncols=(2, math.ceil(num_plot / 2)),
            axes_pad=(0.05 if single_color_bar else 0.5, 0.5),
            share_all=not single_color_bar,
            label_mode="L",
            cbar_location="right",
            cbar_mode="single" if single_color_bar else "each",
        )

        num_client = weights_stats.shape[0]

        for idx, ax in enumerate(grid):
            if idx >= num_plot:
                continue
            if single_round:
                weights = weights_stats[:, plot_rounds[idx]]
            else:
                weights = weights_stats[
                    :, plot_rounds[idx] : plot_rounds[idx + 1]
                ].mean(axis=1)

            if diagonal is not None:
                if diagonal == "zero":
                    np.fill_diagonal(weights, 0)
                else:
                    raise ValueError(f"Unknown diagonal value {diagonal}")

            if single_color_bar:
                im = ax.imshow(weights, vmin=vmin, vmax=vmax)
            else:
                im = ax.imshow(weights)  # , vmin=vmin, vmax=vmax)
            ax.set_title(
                f"Round{' '+ str(plot_rounds[idx]) if single_round else f's {plot_rounds[idx]}-{plot_rounds[idx+1]}'}"
            )
            ax.set_xticks(range(0, num_client), range(1, num_client + 1))
            ax.set_yticks(range(0, num_client), range(1, num_client + 1))

            if not single_color_bar:
                ax.cax.colorbar(im)
        if single_color_bar:
            grid.cbar_axes[0].colorbar(im)

        plt.xlabel(x_label)
        plt.ylabel("Client")
        # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        if self.with_title:
            fig.suptitle(
                f"{name} - {self.config['algo']}-{self.config['dset_name']}/samples per clients: {self.config['samples_per_user']}\nclients:{self.config['num_users']}/label distribution: {self.config['train_label_distribution']}"
            )
        else:
            fig.suptitle(f"{name}")
        single_round_str = "single_round" if single_round else "average"
        name_str = str.lower(name).replace(" ", "_")
        diagonal_str = f"_{diagonal}" if diagonal is not None else ""
        plt.savefig(
            f"{self.plot_dir}/{name_str}_weights_over_time_{single_round_str}{diagonal_str}.png"
        )
        plt.close()

    def plot_clients_score_histogram(self, clients_stats, name):
        # Histogram per clients of mean clients score over rounds
        num_client = clients_stats.shape[0]
        n_col = math.ceil(math.sqrt(num_client))
        n_row = math.ceil(num_client / n_col)
        f, axs = plt.subplots(
            n_row, n_col, figsize=(4 * n_row, 4 * n_col), sharex=True, sharey=True
        )
        for client_idx in range(n_col * n_row):
            ax = plt.subplot(n_row, n_col, client_idx + 1)
            if client_idx < num_client:
                weights = clients_stats[client_idx].mean(axis=0)
                im = plt.bar(range(1, num_client + 1), weights)
                ax.set_title(f"Client {client_idx + 1}")
            ax.set_xticks(range(1, num_client + 1), range(1, num_client + 1))

            label_left = client_idx % n_row == 0
            label_bottom = client_idx >= (n_col - 1) * n_row
            ax.tick_params(bottom=label_bottom, left=label_left)
        if self.with_title:
            f.suptitle(
                f"{name} -- {self.config['algo']}-{self.config['dset_name']}-clients:{self.config['num_users']}-samples per clients: {self.config['samples_per_user']}\nlabel distribution: {self.config['train_label_distribution']})"
            )
        else:
            f.suptitle(f"{name}")
        f.text(0.5, 0.04, name, ha="center", va="center")
        f.text(0.05, 0.5, "Clients", ha="center", va="center", rotation=90)
        plt.savefig(f"{self.plot_dir}/{name.lower().replace(' ', '_')}_hist.png")
        plt.close()

    def plot_score_histogram_evolution_per_clients(
        self,
        clients_stats,
        name,
        num_plot_per_clients=10,
        round_log=False,
        single_round=True,
    ):
        # Histogram per clients of mean clients score over rounds

        num_round = clients_stats.shape[1]
        if round_log:
            plot_rounds = np.logspace(
                1,
                np.log10(num_round),
                num_plot_per_clients + (0 if single_round else 1),
                dtype=int,
            )
            plot_rounds = list(dict.fromkeys(plot_rounds))  # remove duplicates
        else:
            plot_rounds = np.linspace(
                1,
                num_round + 1,
                num_plot_per_clients + (0 if single_round else 1),
                dtype=int,
            )

        num_client = clients_stats.shape[0]
        n_col = num_plot_per_clients
        n_row = num_client
        f, axs = plt.subplots(
            n_row,
            n_col,
            figsize=(3 * n_row + 3, 3 * n_col + 3),
            sharex=True,
            sharey=True,
        )
        for plt_idx in range(n_col * n_row):
            ax = plt.subplot(n_row, n_col, plt_idx + 1)
            client_idx = plt_idx // n_col
            if single_round:
                round_idx = plot_rounds[plt_idx % n_col]
                weights = clients_stats[client_idx, round_idx - 1]
                ax.set_title(f"Client {client_idx + 1}, round {round_idx}")

            else:
                round_start = plot_rounds[plt_idx % num_plot_per_clients]
                round_end = plot_rounds[plt_idx % num_plot_per_clients + 1]
                weights = clients_stats[
                    client_idx, round_start - 1 : round_end - 1
                ].mean(axis=0)
                ax.set_title(
                    f"Client {client_idx + 1}, rounds {round_start}-{round_end}"
                )

            im = plt.bar(range(1, num_client + 1), weights)
            ax.set_xticks(range(1, num_client + 1), range(1, num_client + 1))

            label_left = plt_idx % n_row == 0
            label_bottom = plt_idx >= (n_col - 1) * n_row
            ax.tick_params(bottom=label_bottom, left=label_left)
        if self.with_title:
            f.suptitle(
                f"{name} -- {self.config['algo']}-{self.config['dset_name']}-clients:{self.config['num_users']}-samples per clients: {self.config['samples_per_user']}\nlabel distribution: {self.config['train_label_distribution']})"
            )
        else:
            f.suptitle(f"{name}")
        f.text(0.5, 0.04, name, ha="center", va="center")
        f.text(0.05, 0.5, "Clients", ha="center", va="center", rotation=90)
        plt.savefig(
            f"{self.plot_dir}/{name.lower().replace(' ', '_')}_evolution_hist.png"
        )
        plt.close()

    def plot_clients_collaboration_evolution_separate_plots(self, collab_weights_stats):
        dir = "collab_weights_evol"
        path = self.plot_dir + "/" + dir
        if not os.path.exists(path) or not os.path.isdir(path):
            os.makedirs(path)
        for client in range(collab_weights_stats.shape[0]):
            self.plot_client_collaboration_evolution(collab_weights_stats, client, dir)

    def plot_client_collaboration_evolution(
        self, collab_weights_stats, client_idx, dir=""
    ):
        client_weight_per_rounds = collab_weights_stats[client_idx].T
        # Plot evolution of client weights over round in one graph
        plt.figure()
        x_axis = np.array(range(client_weight_per_rounds.shape[1]))
        for client in range(client_weight_per_rounds.shape[0]):
            plt.plot(
                x_axis, client_weight_per_rounds[client], label=f"client{client+1}"
            )
        plt.legend()
        plt.xlabel("Round")
        plt.ylabel("Collaboration weight")
        if self.with_title:
            plt.title(
                f"Collaboration weight of client {client_idx + 1}- {self.config['algo']}-{self.config['dset_name']}-samples per clients: {self.config['samples_per_user']}\nlabel distribution: {self.config['train_label_distribution']}"
            )
        else:
            plt.title(f"Collaboration weight of client {client_idx + 1}")
        plt.savefig(f"{self.plot_dir}/{dir}/cw_evol_client_{client_idx+1}.png")
        plt.close()

    def plot_clients_weights_evolution_one_plot(self, weights_stats, name):

        num_users, num_rounds, num_score = weights_stats.shape
        grid_size = math.ceil(math.sqrt(num_users))

        # Plot grid of subplot each grid show evolution of client weights over
        # round
        fig, ax = plt.subplots(
            grid_size, grid_size, figsize=(12, 10), sharex=True, sharey=True
        )

        last_ax = ax[0]
        for client in range(num_users):
            row = client // grid_size
            col = client % grid_size
            for c in range(num_score):
                ax[row, col].plot(weights_stats[client, :, c], label=f"client {c+1}")
            ax[row, col].set_title(f"Client {client+1}")
            last_ax = ax[row, col]

        handles, labels = last_ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="center right")
        fig.supxlabel("Round")
        fig.supylabel(f"{name} weight")
        if self.with_title:
            fig.suptitle(
                f"{name} weight of client - {self.config['algo']}-{self.config['dset_name']}-samples per clients: {self.config['samples_per_user']}\nlabel distribution: {self.config['train_label_distribution']}"
            )
        else:
            fig.suptitle(f"{name} weight of client")
        plt.savefig(
            f"{self.plot_dir}/{str.lower(name).replace(' ', '_')}_weights_evolution.png"
        )
        plt.close()

    def plot_experiments_stats(self, stats):
        round_step = stats["round_step"]
        single_score_stats = [
            "train_loss",
            "train_acc",
            "validation_loss",
            "validation_acc",
            "test_acc",
            "test_acc_before_training",
            "test_acc_after_training",
            "lowest_inv_loss",
            "pseudo grad norm",
        ]

        sim_metric = [
            "KL divergence CTCR",
            "Euclidean distance",
            "KL divergence CTAR",
            "Train loss LR",
            "Euclidean distance CTCR KL",
            "KL divergence LTLR",
            "KL divergence CTLR",
            "Dist Correlation AR",
            "Dist Correlation LR",
        ]

        for stat_name, stat in stats.items():
            if stat_name == "round_step":
                continue
            elif stat_name in single_score_stats:
                self.plot_clients_avg_stats_per_round(stat, round_step, stat_name)
                self.plot_clients_stats_per_round(stat, round_step, stat_name)
            else:
                print("Plotting", stat_name)

                if stat_name == "inv class distribution":
                    x_label = "Class"
                else:
                    x_label = "Client"

                self.plot_avg_clients_weights_heatmap(stat, stat_name, x_label=x_label)
                self.plot_avg_clients_weights_heatmap(
                    stat, stat_name, x_label=x_label, diagonal="zero"
                )

                self.plot_clients_weights_heatmap_over_time(
                    stat, stat_name, single_round=True, x_label=x_label
                )  # , plot_rounds=[1, 2, 3, 4, 5, 20, 40, 60, 80, 100])
                self.plot_clients_weights_heatmap_over_time(
                    stat, stat_name, single_round=False, x_label=x_label
                )
                self.plot_clients_weights_heatmap_over_time(
                    stat,
                    stat_name,
                    single_round=False,
                    x_label=x_label,
                    diagonal="zero",
                )
                self.plot_clients_weights_evolution_one_plot(stat, stat_name)

                if isinstance(self.config["dset"], dict) and stat_name in sim_metric:
                    clients_per_community = self.config["num_users"] // len(
                        set(self.config["dset"].values())
                    )
                    self.plot_similarity_metric_accuracy(
                        stat, 1, True, clients_per_community, stat_name
                    )
                    self.plot_similarity_metric_accuracy(
                        stat,
                        clients_per_community - 1,
                        True,
                        clients_per_community,
                        stat_name,
                    )

        stat_name = "Selection probability"
        if stat_name in stats:
            stat = stats[stat_name]
            self.plot_clients_score_histogram(stat, stat_name)
            # self.plot_score_histogram_evolution_per_clients(stat, stat_name, single_round=False)

    def plot_training_distribution(self, split_labels, dset_name, clients_id):
        cls = np.unique(split_labels)
        n_cls = cls.shape[0]

        x = [[str(i)] * n_cls for i in clients_id]
        y = [cls for _ in clients_id]
        s = []
        for clt_idx, clt_id in enumerate(clients_id):
            labels = [np.where(split_labels[clt_idx] == i)[0].shape[0] for i in cls]
            s.append(np.array(labels))

        plt.scatter(x, y, s=s)
        plt.title("Training label distribution for " + dset_name + " dataset")
        plt.xlabel("Client id")
        plt.ylabel("Training labels")
        plt.savefig(f"{self.plot_dir}{dset_name}_training_label_distribution.png")
        plt.close()
        print(f"{self.plot_dir}/{dset_name}_training_label_distribution.png")

    def plot_similarity_metric_accuracy(
        self, similarity_metric, top_k, is_lowest, clients_per_community, name
    ):
        similarity_metric = similarity_metric.copy()
        n_clients = similarity_metric.shape[0]
        n_rounds = similarity_metric.shape[1]
        num_commu = n_clients // clients_per_community

        for i in range(n_rounds):
            np.fill_diagonal(
                similarity_metric[:, i, :], -np.inf if is_lowest else np.inf
            )

        commu_prediction = (similarity_metric.argsort(axis=2) // clients_per_community)[
            :, :, 1:
        ]

        own_cluster_prediction = commu_prediction[:, :, :top_k]
        top_true = (
            np.repeat(np.arange(num_commu), clients_per_community).reshape((1, -1)).T
        )
        true_cluster = np.repeat(top_true, top_k, axis=1)[:, np.newaxis, :]

        cluster_accuracy = np.mean(own_cluster_prediction == true_cluster, axis=(0, 2))
        cluster_accuracy_std = np.std(
            np.mean(own_cluster_prediction == true_cluster, axis=2), axis=0
        )

        plt.plot(cluster_accuracy)
        plt.fill_between(
            np.arange(n_rounds),
            cluster_accuracy - cluster_accuracy_std,
            cluster_accuracy + cluster_accuracy_std,
            alpha=0.2,
        )
        plt.legend()
        plt.title(f"{name} top k accuracy")
        plt.xlabel("Round")
        plt.savefig(f"{self.plot_dir}/top{top_k}_similarity_accuracy.png")
        plt.close()
