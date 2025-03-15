# Image Classification

## Overview

Our environment supports image classification tasks using various ResNet architectures. We provide implementations of ResNet models tailored for different image classification tasks across several datasets. ResNet (Residual Network) models are widely recognized for their effectiveness in deep learning, particularly in image recognition tasks. In this project, we utilize ResNet models to classify images from multiple datasets, including DomainNet, Camelyon17, Digit-Five, CIFAR-10, CIFAR-100, and Medical MNIST. The implementation is designed to handle decentralized machine learning scenarios, allowing multiple users to train a shared model while keeping their data localized.

### Credit:

Credit to Huawei Technologies Co., Ltd. <foss@huawei.com> for ResNet. Taken from [Huawei ResNet implementation](https://github.com/huawei-noah/Data-Efficient-Model-Compression/blob/master/DAFL/resnet.py) for comparison with DAFL.

## Dataset Preparation

We use several datasets for image classification tasks, including DomainNet, Camelyon17, Digit-Five, CIFAR-10, CIFAR-100, and Medical MNIST. Each dataset has specific characteristics and is used for different types of classification tasks. Follow the steps below to download and prepare the datasets:

1. Download the respective datasets from their official sources:
   - **DomainNet**: [DomainNet Dataset page](http://ai.bu.edu/M3SDA/)
   - **Camelyon17**: [Camelyon17 Dataset page](https://camelyon17.grand-challenge.org/)
   - **Digit-Five**: Collection of five digit datasets, including [MNIST](http://yann.lecun.com/exdb/mnist/), [SVHN](http://ufldl.stanford.edu/housenumbers/), [USPS](https://github.com/keras-team/keras/blob/master/keras/datasets/usps.py), [SYN](https://github.com/gabrieleilertsen/unsupervised-mnist/tree/master/unsupervised_mnist/syn), [MNIST-M](https://github.com/NaJaeMin92/MNIST-M)
   - **CIFAR-10** and **CIFAR-100**: [CIFAR Dataset page](https://www.cs.toronto.edu/~kriz/cifar.html)
   - **Medical MNIST**: [Medical MNIST Dataset page](https://www.kaggle.com/andrewmvd/medical-mnist)

## Configure the Training

To set up the training environment, follow these instructions:

1. Install Dependencies: If you haven't already, run `pip install -r requirements.txt`.
2. Configure the system settings. In `src/configs/sys_config.py`, create a system config object such as the example below, with your desired settings.

```
image_classification_system_config = {
    "num_users": 4,
    "dset": "cifar10",
    "dump_dir": "./expt_dump/", # the path to place the results
    "dpath": "./datasets/imgs/cifar10/", # the location of the dataset
    "device_ids": {"node_0": [0], "node_1": [0], "node_2": [0], "node_3": [0]},
    "samples_per_user": 1000,
    "train_label_distribution": "iid",
    "test_label_distribution": "iid",
    "folder_deletion_signal_path":"./expt_dump/folder_deletion.signal"
}
```

3. Configure the algorithm settings. In `src/configs/algo_config.py`, create an algo config object such as the example below, with your desired algorithm.

```
fedavg_image_classify = {
    "algo": "fedavg", # choose any algorithm we support
    "exp_id": "image_classification",
    # Learning setup
    "epochs": 20,
    "model": "resnet18",
    "model_lr": 1e-3,
    "batch_size": 64,
}
```

4. Initiate Training: `mpirun -n 5 python3 main.py`

> _Note: the `-n` flag should be followed by (number of desired users + 1), for the server node._

> The training will proceed across the users as configured. Monitor printed or saved logs to track progress.

> Your result will be written into the `dump_dir` path specified in `sys_config.py`.

## Additional Notes

- Ensure that the setup is correctly configured to avoid issues with client-server communication.
- If you encounter any issues or have suggestions, please open an issue on our [GitHub repository](https://github.com/redacted/sonar).
