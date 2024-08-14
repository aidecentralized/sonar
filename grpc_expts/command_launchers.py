import subprocess
import shlex
import time


def start_server(trial_idx):
    """Start the server process for a specific trial."""
    print(f"Starting the server for trial {trial_idx}...")
    server_process = subprocess.Popen(["python", "server_benchmark.py"])
    return server_process


def launch_clients(trial_commands):
    """Launch client processes in parallel for each trial."""
    client_processes = []
    for cmd in trial_commands:
        client_process = subprocess.Popen(shlex.split(cmd))
        client_processes.append(client_process)
    return client_processes


def wait_for_processes(processes):
    """Wait for all processes to complete."""
    for process in processes:
        process.wait()


def local_sequential_launcher(commands):
    trials = {}
    for cmd in commands:
        split_cmd = cmd.split("--trial_seed ")
        if len(split_cmd) < 2:
            raise ValueError(f"Command does not contain --trial_seed: {cmd}")
        trial_idx = int(split_cmd[1].split(" ")[0])
        if trial_idx not in trials:
            trials[trial_idx] = []
        trials[trial_idx].append(cmd)

    for trial_idx, trial_commands in trials.items():
        # Start the server for the trial
        server_process = start_server(trial_idx)
        time.sleep(5)  # Wait for the server to start

        # Launch client processes
        client_processes = launch_clients(trial_commands)

        # Wait for all client processes to complete
        wait_for_processes(client_processes)

        # Terminate the server
        server_process.terminate()
        server_process.wait()


REGISTRY = {
    "local_sequential": local_sequential_launcher,
}
