# Project SONAR - Self-Organizing Network of Aggregated Representations
A collaborative learning project where users self-organize to improve their ML models by sharing representations of their data or model.

## Main
The application currently uses MPI and GRPC (experimental) to enable communication between different nodes in the network. The goal of the framework to organize everything in a modular manner. That way a researcher or engineer can easily swap out different components of the framework to test their hypothesis or a new algorithm. Most of the important components are specified in the config file which we discuss below: