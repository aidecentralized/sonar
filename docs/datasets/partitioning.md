In decentralized collaborative learning, data partitioning is the process of dividing the data across the participating nodes. Over the last few years, several data partitioning strategies have been proposed to model realistic scenarios and improve the performance of decentralized learning systems. In this section, we discuss some of the data partitioning strategies supported by our framework.

## IID Data Partitioning

In IID (Independent and Identically Distributed) data partitioning, the data is divided uniformly across the nodes. Each node receives a subset of the data, and the data distribution across the nodes is identical. This partitioning strategy is one of the most simple and widely used strategies when simulating decentralized learning scenarios. However, in real-world scenarios, IID partitioning is rarely observed, as data is often non-IID due to variations in data collection processes, data sources, and data distributions.

## Non-IID Data Partitioning

Non-IID data partitioning refers to a set of strategies where the data distribution across the nodes is non-uniform. In non-IID partitioning, each node receives a distinct subset of the data, and the data distribution across the nodes is heterogeneous. Our framework supports several non-IID partitioning strategies, including:

### Label Distribution

In label distribution partitioning, the data is divided based on the distribution over the labels. We model the non-IID partitioning for labels in two ways - 1. **Distinct Labels**: Each node receives a distinct set of labels, and 2. **Imbalanced Labels**: Nodes receive imbalanced label distributions, where some labels are over-represented compared to others. This is modeled using Dirichlet distribution.

### Domain Distribution

In domain distribution partitioning, the data is divided based on the distribution over the domains. Each node receives a distinct subset of the domains, and the data distribution across the nodes is heterogeneous. This partitioning strategy is useful when the data is collected from multiple sources or domains, and each node represents a distinct domain. For example, in the DomainNet dataset, each node can represent a different domain such as sketch, real image, quickdraw, painting, infograph, or clipart.

### Multi-task Learning

In multi-task learning partitioning, the data is divided based on multiple tasks or objectives. Each node receives a distinct subset of tasks, and the data distribution across the nodes is heterogeneous. For example, in Medical MNIST dataset, each node can focus on a specific medical condition such as pneumonia, emphysema, or cardiomegaly.

### Feature Distribution

In feature distribution partitioning, the data is divided based on the distribution over the features. Each node receives a distinct subset of features, and the data distribution across the nodes is heterogeneous. For example, in CIFAR-10 dataset, we simulate this scenario by rotating the images and dividing them across nodes based on the rotation angle.