# SONAR: Self-Organizing Network of Aggregated Representations
Project by MIT Media Lab
A collaborative learning project where users self-organize to improve their ML models by sharing representations of their data or model.

## Main
The application currently uses MPI and GRPC (experimental) to enable communication between different nodes in the network. The goal of the framework to organize everything in a modular manner. That way a researcher or engineer can easily swap out different components of the framework to test their hypothesis or a new algorithm.

| Topology                   | Train                  |                     |                     | Test                   |                     |                     |
|----------------------------|------------------------|---------------------|---------------------|------------------------|---------------------|---------------------|
|                            | 1                      | 2                   | 3                   | 1                      | 2                   | 3                   |
| Isolated                   | 208.0<sub>(0.3)</sub>  | 208.0<sub>(0.3)</sub>| 208.0<sub>(0.0)</sub>| 44.5<sub>(6.9)</sub>  | 44.5<sub>(6.9)</sub>| 44.5<sub>(6.9)</sub>|
| Central*                   | 208.5<sub>(0.1)</sub>  | 208.5<sub>(0.0)</sub>| 208.5<sub>(0.0)</sub>| 33.97<sub>(14.2)</sub>| 33.97<sub>(14.2)</sub>| 33.97<sub>(14.2)</sub>|
| Random                     | 205.4<sub>(1.0)</sub>  | 205.5<sub>(0.9)</sub>| 205.9<sub>(0.8)</sub>| 54.9<sub>(5.3)</sub>  | 56.0<sub>(5.8)</sub>| 56.2<sub>(5.6)</sub>|
| Ring                       | 198.8<sub>(3.1)</sub>  | 198.7<sub>(3.3)</sub>| 198.7<sub>(3.4)</sub>| 47.8<sub>(7.3)</sub>  | 46.9<sub>(6.9)</sub>| 47.6<sub>(7.1)</sub>|
| Grid                       | 202.6<sub>(1.5)</sub>  | 203.9<sub>(1.4)</sub>| 204.5<sub>(1.3)</sub>| 49.3<sub>(6.0)</sub>  | 48.8<sub>(6.0)</sub>| 48.1<sub>(6.1)</sub>|
| Torus                      | 202.0<sub>(1.2)</sub>  | 203.2<sub>(1.2)</sub>| 204.0<sub>(1.3)</sub>| 50.2<sub>(6.0)</sub>  | 50.7<sub>(6.6)</sub>| 50.3<sub>(6.2)</sub>|
| Similarity based (top-k)   | 206.4<sub>(1.6)</sub>  | 197.6<sub>(7.3)</sub>| 200.4<sub>(4.4)</sub>| 47.3<sub>(8.3)</sub>  | 48.4<sub>(8.5)</sub>| 52.8<sub>(7.2)</sub>|
| Swarm                      | 183.2<sub>(3.5)</sub>  | 183.1<sub>(3.6)</sub>| 183.2<sub>(3.6)</sub>| 52.2<sub>(8.7)</sub>  | 52.3<sub>(8.7)</sub>| 52.4<sub>(8.6)</sub>|
| L2C                        | 167.0<sub>(25.4)</sub> | 158.8<sub>(30.6)</sub>| 152.8<sub>(35.2)</sub>| 37.6<sub>(7.4)</sub>  | 36.6<sub>(7.4)</sub>| 35.8<sub>(7.7)</sub>|

**Table 1:** Performance overview (AUC) of various topologies with different number of collaborators.


### Running the code
Let's say you want to run the model training of 3 nodes on a machine. That means there will be 4 nodes in total because there is 1 more node in addition to the clients --- server.
The whole point of this project is to eventually transition to a distributed system where each node can be a separate machine and a server is simply another node. But for now, this is how things are done.
You can do execute the 3 node simulation by running the following command:
`mpirun -np 4 -host localhost:11 python main.py`

### Config file
The config file is the most important file when running the code. The current set up combines a system config with an algorithm config file. Always be sure of what config you are using. We have intentionally kept configuration files as a python file which is typically a big red flag in software engineering. But we did this because it enables plenty of quick automations and flexibility. Be very careful with the config file because it is easy to overlook some of the configurations such as device ids, number of clients etc.

### Reproducability
One of the awesome things about this project is that whenever you run an experiment, all the source code, logs, and model weights are saved in a separate folder. This is done to ensure that you can reproduce the results by looking at the code that was responsible for the results. The naming of the folder is based on the keys inside the config file. That also means you can not run the same experiment again without renaming/deleting the previous experimental run. The code automatically asks you to press `r` to remove and create a new folder. Be careful you are not overwriting someone else's results.

### Logging
We log the results in the console and also in a log file that captures the same information. We also log a few metrics for the tensorboard. The tensorboard logs can be viewed by running tensorboard as follows:
`tensorboard --logdir=expt_dump/ --host 0.0.0.0`. Assuming `expt_dump` is the folder where the experiment logs are stored.

### Open Projects and Tasks
- Experiments: Topology Experiments
- Benchmark Redesign: Improve telemetry, Seperate Topology
- Comprehensive Documentation

## Set up MK Docs on Local
* Clone the repository
* `pip install mkdocs-material`
* `mkdocs build`
* Add `repo_url: https://github.com/aidecentralized/sonar/` to `mkdocs.yml`
*  `mkdocs serve`
