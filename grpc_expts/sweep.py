import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import numpy as np
import tqdm
import shlex
from hparams_registry import random_hparams, default_hparams
from command_launchers import REGISTRY

BASE_DIR = "/projectnb/ds598/projects/xthomas/temp/sonar/grpc_expts"
LOG_DIR = f"{BASE_DIR}/sweep_logs"


class Job:
    NOT_LAUNCHED = "Not launched"
    INCOMPLETE = "Incomplete"
    DONE = "Done"

    def __init__(self, hparams, train_args_list, sweep_output_dir):
        # Generate hash based on hyperparameters only
        hparams_str = json.dumps(hparams, sort_keys=True)
        hparams_hash = hashlib.md5(hparams_str.encode("utf-8")).hexdigest()
        self.output_dir = os.path.join(sweep_output_dir, hparams_hash)

        self.train_args_list = train_args_list
        self.commands = []
        self.state = Job.DONE
        for train_args in train_args_list:
            trial_output_dir = os.path.join(
                self.output_dir,
                f'trial_{train_args["trial_seed"]}',
                f'client_{train_args["client_id"]}',
            )
            train_args["output_dir"] = trial_output_dir

            if not os.path.exists(trial_output_dir):
                self.state = Job.NOT_LAUNCHED
            elif not os.path.exists(os.path.join(trial_output_dir, "done")):
                self.state = Job.INCOMPLETE

            command = ["python", "client_benchmark.py"]
            for k, v in sorted(train_args.items()):
                if isinstance(v, dict):
                    v = json.dumps(v)
                    command.append(
                        f"--{k} {shlex.quote(v)}"
                    )  # Properly escape JSON string
                elif isinstance(v, list):
                    v = " ".join([str(v_) for v_ in v])
                    command.append(f"--{k} {v}")
                elif isinstance(v, str):
                    v = shlex.quote(v)
                    command.append(f"--{k} {v}")
                else:
                    command.append(f"--{k} {v}")
            self.commands.append(" ".join(command))

    def __str__(self):
        job_info = [
            (
                args["dataset"],
                args["algorithm"],
                args["client_id"],
                args["hparams_seed"],
                args["trial_seed"],
            )
            for args in self.train_args_list
        ]
        hyperparams = [
            {
                k: v
                for k, v in args.items()
                if k
                not in [
                    "dataset",
                    "algorithm",
                    "client_id",
                    "hparams_seed",
                    "trial_seed",
                ]
            }
            for args in self.train_args_list
        ]
        return "{}: {} {} {}".format(self.state, self.output_dir, job_info, hyperparams)

    @staticmethod
    def launch(jobs, launcher_fn):
        print("Launching...")
        np.random.shuffle(jobs)

        # Create directories only during launch
        for job in tqdm.tqdm(jobs, leave=False):
            for command in job.commands:
                trial_output_dir = command.split("--output_dir ")[1].split(" ")[0]
                os.makedirs(trial_output_dir, exist_ok=True)

        for job in jobs:
            print(f"Launching job with output_dir: {job.output_dir}")
            launcher_fn(job.commands)

        print(f"Launched {len(jobs)} jobs!")

    @staticmethod
    def delete(jobs):
        print("Deleting...")
        for job in jobs:
            shutil.rmtree(job.output_dir)
        print(f"Deleted {len(jobs)} jobs!")


def make_args_list(
    num_clients,
    num_hparams,
    num_trials,
    host,
    model_arch,
    dataset,
    algorithm,
    hparams_seed,
):
    jobs = []
    for hparam_seed in range(num_hparams):
        if hparam_seed == 0:
            hparams = default_hparams(algorithm, dataset)
        else:
            hparams = random_hparams(algorithm, dataset, hparam_seed)

        train_args_list = []
        for trial_seed in range(num_trials):
            for client_id in range(num_clients):
                train_args = {
                    "host": host,
                    "model_arch": model_arch,
                    "dataset": dataset,
                    "algorithm": algorithm,
                    "client_id": client_id,
                    "hparams": hparams,
                    "hparams_seed": hparam_seed,
                    "trial_seed": trial_seed,
                }
                train_args_list.append(train_args)
        jobs.append(Job(hparams, train_args_list, LOG_DIR))
    return jobs


def ask_for_confirmation():
    response = input("Are you sure? (y/n) ")
    if not response.lower().strip()[:1] == "y":
        print("Nevermind!")
        exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a sweep")
    parser.add_argument(
        "command", choices=["launch", "delete_incomplete", "delete_all"]
    )
    parser.add_argument("--host", type=str, default="scc-204.scc.bu.edu:50051")
    parser.add_argument("--model_arch", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--algorithm", type=str, default="fedavg")
    parser.add_argument("--num_clients", type=int, default=2)
    parser.add_argument("--num_hparams", type=int, default=5)
    parser.add_argument("--num_trials", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default=LOG_DIR)
    parser.add_argument("--hparams_seed", type=int, default=0)
    parser.add_argument(
        "--command_launcher",
        type=str,
        choices=REGISTRY.keys(),
        default="local_sequential",
    )
    parser.add_argument("--skip_confirmation", action="store_true")
    args = parser.parse_args()

    jobs = make_args_list(
        num_clients=args.num_clients,
        num_hparams=args.num_hparams,
        num_trials=args.num_trials,
        host=args.host,
        model_arch=args.model_arch,
        dataset=args.dataset,
        algorithm=args.algorithm,
        hparams_seed=args.hparams_seed,
    )

    for job in jobs:
        print(job)
        print()
    print(
        "\n{} jobs: {} done, {} incomplete, {} not launched.\n".format(
            len(jobs),
            len([j for j in jobs if j.state == Job.DONE]),
            len([j for j in jobs if j.state == Job.INCOMPLETE]),
            len([j for j in jobs if j.state == Job.NOT_LAUNCHED]),
        )
    )

    if args.command == "launch":
        to_launch = [j for j in jobs if j.state == Job.NOT_LAUNCHED]
        print(f"About to launch {len(to_launch)} jobs.")
        if not args.skip_confirmation:
            ask_for_confirmation()
        launcher_fn = REGISTRY[args.command_launcher]
        Job.launch(to_launch, launcher_fn)

    elif args.command == "delete_incomplete":
        to_delete = [j for j in jobs if j.state == Job.INCOMPLETE]
        print(f"About to delete {len(to_delete)} jobs.")
        if not args.skip_confirmation:
            ask_for_confirmation()
        Job.delete(to_delete)

    elif args.command == "delete_all":
        print(f"About to delete all {len(jobs)} jobs.")
        if not args.skip_confirmation:
            ask_for_confirmation()
        Job.delete(jobs)
