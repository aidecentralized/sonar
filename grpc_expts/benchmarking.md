## Benchmarking Docs (WIP)

### `sweep.py`

The sweep.py script is designed to automate the process of running hyperparameter sweep jobs for the benchmark tasks. It allows users to launch multiple experiments with varying hyperparameter sets, manage and monitor their states, and clean up by deleting incomplete or all jobs as needed. This script ensures efficient and organized execution of large-scale experimentation by creating unique directories for each set of hyperparameters and managing the execution of client and server processes.

It facilitates the following tasks:

1) Launching hyperparameter sweep jobs: Runs multiple machine learning experiments with different sets of hyperparameters.
2) Deleting incomplete jobs: Removes jobs that were not completed successfully.
3) Deleting all jobs: Removes all jobs, regardless of their state.

Job:

A Job represents a set of training tasks associated with a specific configuration of hyperparameters. Each job can have multiple trials and clients. The Job class manages the state of these tasks, organizes the necessary directories, and constructs commands to execute the benchmark script. It has the following key components:

State Management: Tracks whether a job is not launched, incomplete, or done.
Initialization: Takes a list of training arguments and prepares the output directory structure.
Command Construction: Builds the command to run the benchmark script with the specified hyperparameters.
Directory Handling: Ensures that necessary directories are created during the job launch.
Static Methods:
launch: Launches a list of jobs using a specified launcher function.
delete: Deletes the directories of specified jobs.
The Job class encapsulates the details of running an experiment with a particular set of hyperparameters, making it easier to manage and execute large-scale hyperparameter sweeps efficiently.

### `hparams_registry.py`

The `hparams_registry.py` module provides functions to define and generate hyperparameters for machine learning algorithms and datasets. It includes mechanisms to create default and random hyperparameter sets, ensuring reproducibility and variability in hyperparameter sweeps. By using a combination of fixed defaults and randomized values based on a seed, this module facilitates systematic experimentation and performance evaluation of different algorithmic configurations.  Additionally, it allows for the dynamic addition of hyperparameters specific to certain algorithms and datasets, enhancing flexibility and customization in experiments.

### `command_launchers.py`

The `command_launchers.py` module provides functions to launch client and server processes for hyperparameter sweep jobs. It includes a method to sequentially run client commands for each trial while starting a new server for each trial. This ensures that each trial runs in isolation, facilitating systematic experimentation and performance evaluation. In essence, it starts a new server for each job and trial, ensuring isolation and preventing conflicts between different trials, thereby maintaining the integrity and independence of each experimental run.

### Steps in the Workflow

1. **Start**: Begin the execution of the `sweep.py` script.
2. **Parse Command-Line Arguments**: 
   - The script reads and parses the provided command-line arguments.
3. **Generate Hyperparameter Sets**: 
   - Hyperparameter sets are generated based on the specified arguments.
   - The `hparams_registry` module is used to create default and random hyperparameter sets.
4. **Create Jobs**: 
   - Jobs are created for each set of hyperparameters by initializing the `Job` class with the training arguments and output directories.
5. **Check Job States**: 
   - The script checks the state of each job (Not launched, Incomplete, Done) by examining if the output directories and done files exist.
   - Jobs are printed with their current state.
6. **Ask for Confirmation**: 
   - If there are jobs to be launched, the script asks for user confirmation unless the `--skip_confirmation` flag is set.
7. **Confirmation Received?**: 
   - The script proceeds based on the user's response.
   - **Yes**: Continue to create directories for the jobs.
     - **No**: Exit the script.
8. **Create Directories for Jobs**: 
   - Directories for the jobs are created if they do not already exist.
9. **Launch Server for Each Trial**: 
    - A new server is started for each trial using the `subprocess.Popen` command.
    - The server is launched by running `server_benchmark.py`.
10. **Launch Client Processes for Each Trial**: 
    - Client processes are launched in parallel for each trial using `subprocess.Popen`.
    - Each client runs the `client_benchmark.py` script with the specified arguments.
11. **Wait for All Client Processes to Complete**: 
    - The script waits for all client processes to finish using `client_process.wait()`.
12. **Terminate Server**: 
    - The server is terminated after the client processes complete using `server_process.terminate()`.
13. **Mark Job as Complete**: 
    - The job is marked as complete by creating a `done` file in the output directory.
14. **All Jobs Complete**: 
    - All jobs have been processed.
15. **End**: 
    - The script execution ends.

### Example Command

Run the bash script:

```sh
bash run_sweep.sh
```

### Example Results:

Best hyperparameter setting: db339c756bf05546161ab518b8b50bfb with mean test accuracy: 0.7795    
Model: resnet18    
Dataset: cifar10    
Algorithm: Fedavg    
Best Test Accuracy: 0.7795 ± 0.1053    

Table:

| Model    | Dataset   | Algorithm   | Best Test Accuracy   |
|----------|-----------|-------------|----------------------|
| resnet18 | cifar10   | Fedavg      | 0.7795 ± 0.1053      |

Job Statistics:

Number of hyperparameter sets: 3    
Number of trials: 3    