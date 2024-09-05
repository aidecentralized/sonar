## Comparison with Existing FL Frameworks

There are several frameworks available for federated learning, each with its unique features and capabilities. However, the focus of our work is on decentralized and large scale collaboration. Therefore, we identify the following five key features that are essential for evaluating the capabilities of federated learning frameworks in the context of decentralized collaborative learning:

1. **Decentralized Training**: The ability to train models in a decentralized manner, where the training does not necessarily depend upon a central server or a single coordinator.

2. **Topology**: Support for a wide range of network topologies, including gossip-based, ring, tree, mesh, data-driven, etc. If a framework supports these multiple topologies out of the box, then it becomes easier to rapidly prototype and experiment with different network configurations.

3. **Data Partitioning**: The ability to handle different data partitioning strategies, such as IID, non-IID, personalized, and multi-task learning. This is crucial for modeling real-world scenarios where data is distributed across multiple users in a non-uniform manner.

4. **Privacy & Robustness**: Support for privacy-preserving techniques such as differential privacy, secure aggregation, etc., to model privacy concerns in collaborative learning settings. Additionally, a framework should provide testbeds for empirically evaluating the privacy by applying reconstruction or membership inference attacks. From a robustness perspective, the framework should be able to simulate malicious behavior, such as Byzantine attacks, stragglers, and model poisoning etc and provide off-the-shelf support for mechanisms to mitigate these attacks.

5. **Collaborator Selection**: The ability to select collaborators dynamically based on various criteria such as network conditions, data distribution, model performance, etc. This is essential for large-scale decentralized collaboration where the network topology is dynamic and collaborators may join or leave the network at any time.


### Scoring Mechanism
We compare the capabilities of existing federated learning frameworks with CollaBench based on the above features. We use a simple scoring mechanism to evaluate the capabilities of each framework. The score ranges from 0 to 2, where 0 indicates no support, 1 indicates partial support, and 2 indicates full support for the feature. Therefore, we get a total score out of 10 for each framework based on the five key features mentioned above.


| Framework | Decentralized training | Topology | Data Partitioning | Privacy & Robustness | Collaborator Selection | Total Score |
|-----------|----------|----------|----------|----------|----------|-------------|
|   [P2PFL](https://pguijas.github.io/p2pfl/index.html)   | 2 | 1 | 0 | 0 | 0 | 3 |
|   [FedML](https://docs.tensoropera.ai/)   | 2 | 1 | 2 | 1 | 0 | 6 |
|   [EasyFL](https://github.com/EasyFL-AI/EasyFL)  | 0 | 0 | 1 | 0 | 0 | 1 |
|   [COALA](https://github.com/SonyResearch/COALA)   | 0 | 0 | 2 | 0 | 0 | 2 |
|  [FedScale](https://fedscale.ai/) | 1 | 0 | 2 | 0 | 1 | 4 |
|   [Flower](https://flower.ai/docs/framework/index.html)  | 0 | 0 | 2 | 2 | 0 | 4 |
| CollaBench (Ours) | 1 | 2 | 2 | 0 | 2 | 7 |

		
Table 1. Head-to-head comparison of capabilities between existing frameworks and CollaBench. Note that this comparison is purely focused on the use-cases of decentralized collaborative learning and does not reflect the overall capabilities of the frameworks. Some of these frameworks are pretty advanced and have additional features that are not relevant to decentralized collaboration.