# Getting Started

> If you would like to contribute to improving this project, please refer to our issues page where we have our open [issues](github.com/redacted/sonar/issues) in the side navigation bar.

## Installation

- `git clone` the repository
- `pip install -r requirements.txt`

## Running the code

Let's say you want to run the model training of 3 nodes on a machine. That means there will be 4 nodes in total because there is 1 more node in addition to the clients --- server.
The whole point of this project is to eventually transition to a distributed system where each node can be a separate machine and a server is simply another node. But for now, this is how things are done.
You can do execute the 3 node simulation by running the following command:
`mpirun -np 4 -host localhost:11 python main.py`

## Config file

The config file is the most important file when running the code. The current set up combines a system config with an algorithm config file. Always be sure of what config you are using. We have intentionally kept configuration files as a python file which is typically a big red flag in software engineering. But we did this because it enables plenty of quick automations and flexibility. Be very careful with the config file because it is easy to overlook some of the configurations such as device ids, number of clients etc.

## Reproducability

One of the awesome things about this project is that whenever you run an experiment, all the source code, logs, and model weights are saved in a separate folder. This is done to ensure that you can reproduce the results by looking at the code that was responsible for the results. The naming of the folder is based on the keys inside the config file. That also means you can not run the same experiment again without renaming/deleting the previous experimental run. The code automatically asks you to press `r` to remove and create a new folder. Be careful you are not overwriting someone else's results.
