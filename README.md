# Project SONAR - Self-Organizing Network of Aggregated Representations
A collaborative learning project where users self-organize to improve their ML models by sharing representations of their data or model.

## Main
The application currently uses MPI and GRPC (experimental) to enable communication between different nodes in the network. The goal of the framework to organize everything in a modular manner. That way a researcher or engineer can easily swap out different components of the framework to test their hypothesis or a new algorithm.

### Objectives
- [ ] O1: Benchmark existing configurations and algorithms.
- [ ] O2: Separate the communication layer (topology) from collaborative learning algorithms.
- [ ] O3: Implement a few more collaborative learning algorithms.
- [ ] O4: Write the GRPC module for the communication layer. Then we don't need to rely on MPI which requires ssh access to all the nodes.
- [ ] O5: Improve telemetry and logging for visualization of the network.
- [ ] 06: Fault tolerance and rogue clients simulation
- [ ] 07: Comprehensive documentation - https://github.com/squidfunk/mkdocs-material
- [ ] 08: Eliminate the need to add a BaseServer module, keep it backward compatible by instantiating the server as yet another node.
- [ ] 09: Build testing suite
- [ ] 10: Set up milestones for transition to full API like interface and then launch on `pip`
