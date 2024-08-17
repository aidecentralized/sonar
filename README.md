![Warning](https://img.shields.io/badge/Warning-This%20project%20is%20in%20beta-yellow)

# Project SONAR - Self-Organizing Network of Aggregated Representations

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/markdown-guide/badge/?version=latest)](https://aidecentralized.github.io/sonar/)

A collaborative learning project where users self-organize to improve their ML models by sharing representations of their data or model.

⚠️**Warning:** This project is currently under a major revamp and may contain bugs.

## Main
The application currently uses MPI and GRPC (experimental) to enable communication between different nodes in the network. The goal of the framework to organize everything in a modular manner. That way a researcher or engineer can easily swap out different components of the framework to test their hypothesis or a new algorithm.

### Framework Rewrite Objective
- [ ] O1: Benchmark existing configurations and algorithms.
- [x] O2: Separate the communication layer (topology) from collaborative learning algorithms.
- [ ] O3: Implement a few more collaborative learning algorithms.
- [ ] O4: Write the GRPC module for the communication layer. Then we don't need to rely on MPI which requires ssh access to all the nodes. See https://github.com/aidecentralized/sonar/issues/20
- [ ] O5: Improve telemetry and logging for visualization of the network. See https://github.com/aidecentralized/sonar/issues/11
- [ ] O6: Fault tolerance and rogue clients simulation.
- [ ] O7: Eliminate the need to add a BaseServer module, keep it backward compatible by instantiating the server as yet another node.
- [ ] O8: Build testing suite. See https://github.com/aidecentralized/sonar/issues/21
- [ ] O9: Set up milestones for transition to full API like interface and then launch on `pip`
