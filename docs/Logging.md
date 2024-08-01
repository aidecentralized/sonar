# Logging Documentation

This document provides a detailed overview of what is being logged in the Sonar setup. 

## Table of Contents

1. [Overview](#overview)
2. [Logging Types](#logging-types)
3. [Log Sources](#log-sources)
4. [Log Details](#log-details)

## Overview

This documentation aims to provide transparency on the logging mechanisms implemented in the Sonar project. It includes information on the types of data being logged, their sources, formats, and purposes.

## Logging Types

- **DEBUG:** Detailed information, typically of interest only when diagnosing problems.
- **INFO:** Confirmation that things are working as expected.
- **Tensorboard logging**: Logging specific metrics, images, and other data to TensorBoard for visualization and analysis.
    - Console Logging: Logs a message to the console.
    - Scalar Logging: Logs scalar values to TensorBoard for tracking metrics(loss, accuracy) over time.
    - Image Logging: Logs images to both a file and TensorBoard for visual analysis.

## Log Sources

| Component/Module   | Data Logged                                      | Log Level     | Format      | Storage Location                    | Frequency/Trigger                      |
|--------------------|--------------------------------------------------|---------------|-------------|-------------------------------------|----------------------------------------|
| Model Training (FL) | Aggregated model metrics, client updates         | INFO, DEBUG   | Plain text  | `./expt_dump/<experiment_name>/logs/client_<client_index>/summary.txt`     | On every FL round   

## Log Details

### Federated Learning
Logs aggregated model metrics (loss and accuracy) and updates from clients to track the overall progress and performance of the federated learning process. Additionally, logs include training loss and accuracy from individual clients. Also logs communication events between the server and clients to monitor interactions and data exchange.

