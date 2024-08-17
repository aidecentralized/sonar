# Customizability

The framework provides an easy way to customize various aspects such as the dataset, model, and topology. By following these steps, users can switch topologies by simply changing a single parameter in the configuration file.

## 1. Customizing the Dataset:
To customize the dataset, users can modify the data loading code to read data from different sources or apply preprocessing techniques specific to their needs. This can involve changing file paths, data augmentation techniques, or data normalization methods.

### Example 1:

To change the dataset, update the `dpath` parameter in the `sys_config.py` file with the address of your dataset folder.

## 2. Customizing the Model:
Users can easily customize the model by modifying the type of model you want to use.

### Example 2:

To change the model, update the `model` parameter in the `algo_config.py` file. Choose from available options like `resnet10`, `resnet34`, `yolo`, etc.

## 3. Customizing the Topology:
To switch topologies, users can define different topology configurations in the configuration file. By changing a single parameter, the framework will automatically use the corresponding topology during runtime. This allows switching between different network architectures without extensive code modifications. The supported topologies are mentioned in the documentation and new ones are constantly being added.

### Example 3:

In the `algo_config.py` file, select the desired topology for an experiment by changing the `algo` parameter. This simple change allows switching between different topologies.

By modifying a single parameter in the configuration file, multiple experiments can be run.

