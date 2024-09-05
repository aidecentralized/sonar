# Configuration File Overview

## Importance of the Configuration File

The configuration file is the cornerstone of running our codebase, providing the necessary settings to ensure smooth execution. It combines a **system configuration** with an **algorithm configuration** to manage both infrastructure and algorithmic parameters. This separation is crucial for maintaining flexibility and clarity, allowing different aspects of the system to be configured independently.

### Why This Setup?

We intentionally chose to maintain the configuration as a Python file. The Python-based configuration allows for rapid automation and adaptation, enabling researchers to quickly iterate and customize configurations as needed. However, this flexibility comes with responsibility. It is essential to carefully manage the configuration, as it is easy to overlook critical settings such as device IDs or the number of clients. Always double-check your configuration file before running the code to avoid unintended behavior.

## Purpose of Configuration Separation

The configuration files are split into two main parts:

1. **System Configuration (`system_config`)**: Manages the infrastructure-related aspects, such as client configurations, GPU device allocation, and data splits. This ensures that researchers focused on algorithm development can work without needing to adjust system-level details.

2. **Algorithm Configuration (`algo_config`)**: Contains settings specific to the algorithm, such as hyperparameters, learning rates, and model architecture. This configuration is designed to be independent of the system configuration, allowing for modularity and easier experimentation.

These two configurations are combined at runtime in the `scheduler.py` file, which orchestrates the execution of the code based on the provided settings.

## Example Use Case

Consider a scenario where you need to allocate specific GPUs to different clients and define unique data splits for each. These details would be managed in the `system_config`, ensuring that they are isolated from the algorithm's logic. If a researcher wants to test a new optimization algorithm, they can do so by modifying the `algo_config` without worrying about the underlying system setup.

This separation simplifies collaboration and experimentation, allowing different team members to focus on their respective domains without interfering with each other's work.