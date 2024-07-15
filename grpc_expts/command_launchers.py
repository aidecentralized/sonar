import subprocess
import shlex
import time


def local_sequential_launcher(commands):
    for command_group in commands:
        # Group commands by trial index
        trials = {}
        for cmd in command_group:
            trial_idx = int(cmd.split("--trial_seed ")[1].split(" ")[0])
            if trial_idx not in trials:
                trials[trial_idx] = []
            trials[trial_idx].append(cmd)

        for trial_idx, trial_commands in trials.items():
            # Start a new server for each trial
            print(f"Starting the server for trial {trial_idx}...")
            server_process = subprocess.Popen(["python", "server_benchmark.py"])

            # Wait a few seconds to ensure the server has started
            time.sleep(5)

            # Launch the client processes in parallel for each trial
            client_processes = []
            for cmd in trial_commands:
                print(f"Preparing to launch client command: {cmd}")
                client_process = subprocess.Popen(shlex.split(cmd))
                client_processes.append(client_process)
                print(f"Launched client command: {cmd}")

            # Wait for all client processes to complete
            for client_process in client_processes:
                client_process.wait()
                print(f"Client process finished with command: {client_process.args}")

            # Terminate the server
            server_process.terminate()
            server_process.wait()
            print(f"Finished trial {trial_idx} with commands: {trial_commands}")


REGISTRY = {
    "local_sequential": local_sequential_launcher,
}
