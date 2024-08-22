# Multi-agent collaboration with gRPC

In this tutorial, we will discuss how to use gRPC for training models across multiple machines. If you have not already read the [Getting Started](./getting-started.md) guide, we recommend you do so before proceeding.

> **_NOTE:_**  If you are running experiments on a single machine right now then MPI is easier to set up and get started.

## Overview
The main advantage of our abstract communication layer is that the same code runs regardless of the fact you are using MPI or gRPC underneath. As long as the communication layer is implemented correctly, the rest of the code remains the same. This is a huge advantage for the framework as it allows us to switch between different communication layers without changing the code.

## Running the code
Let's say you want to run the decentralized training with 80 users on 4 machines. Our implementation currently requires a coordinating node to manage the orchestration. Therefore, there will be 81 nodes in total. Make sure `sys_config.py` has `num_users: 80` in the config. You should run the following command on all 4 machines:

``` bash
python main_grpc.py -n 20 -host randomhost42.mit.edu
```

On **one** of the machines that you want to use as a coordinator node (let's say it is `randomhost43.mit.edu`), change the `peer_ids` with the hostname and the port you want to run the coordinator node and then run the following command:

``` bash
python main.py -super true
```

> **_NOTE:_**  Most of the algorithms right now do not use the new communication protocol, hence you can only use the old MPI version with them. We are working on updating the algorithms to use the new communication protocol.

## Protocol
1. Each node registers to the coordinating node and receives a unique rank.
2. Each node then tries to find available ports to start a listener.
3. Once a quorum has been reached, the coordinating node sends a message to all nodes with the list of all nodes and their ports.
4. Each node then starts local training.


## FAQ
1. **Is it really decentralized if it requires a coordinating node?**
    - Yes, it is. The coordinating node is required for all nodes to discover each other. In fact, going forward, we plan to provide support for several nodes to act as coordinating nodes. This will make the system more robust and fault-tolerant. We are looking for contributors to implement a distributed hash table (DHT) to make our system more like BitTorrent. So, if you have a knack for distributed systems, please reach out to us.
2. **Why do I need main_grpc.py and main.py?**
    - The main.py file is the actual file. However, if you are running lots of nodes, it is easier to run the main_grpc.py file which will automatically run the main.py file `n` times. This is just a convenience script.
3. **How do I know if the nodes are communicating?**
    - You can check the console logs of the coordinating node. It will print out the messages it is sending and receiving. You can also check the logs of the nodes to see their local training status.
4. **My training is stuck or failed. How do I stop and restart?**
    - GRPC simulation starts a lot of threads and even if one of them fail right now then you will have to kill all of them and start all over. So, here is a command to get the pid of all the threads and kill them all at once:
    ``` bash
    for pid in $(ps aux|grep 'python main.py' | cut -b 10-16); do kill -9 $pid; done
    ```
    Make sure you remove the experiment folder before starting again.
