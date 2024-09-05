# Object Detection

## Overview
Our environment supports object detection and classification tasks. We support an implementation of YOLOv3 objection detection model using the Pascal VOC dataset. The YOLOv3 (You Only Look Once, version 3) model is a state-of-the-art object detection algorithm known for its speed and accuracy. It performs both object detection and classification in a single forward pass through the network, making it highly efficient. In this project, we adapt YOLOv3 to work in a decentralized machine learning setup, which allows multiple users to train a shared model while keeping their data localized.

### Credit
The implementation of YOLOv3 in this project is based on the [GeeksforGeeks YOLOv3 tutorial](https://www.geeksforgeeks.org/yolov3-from-scratch-using-pytorch/). Special thanks to the authors for providing a detailed guide that served as the foundation for this work.

## Dataset Preparation
We use the Pascal VOC dataset, a popular benchmark in object detection tasks. Follow the steps below to download and prepare the dataset:
1) Download Pascal VOC Data:

* Visit the [Pascal VOC Dataset page](http://host.robots.ox.ac.uk/pascal/VOC/).
* Download the VOC 2012 dataset

2) Extract and Structure the Dataset:

* Extract the downloaded dataset into a directory of your choice.
* Ensure the directory structure is as follows:
```
VOCdevkit/
  VOC2012/
    Annotations/
    ImageSets/
    JPEGImages/
    ...
```

3) (Optional) Split the data
* You can split the data according to your desired distribution by labeling a text file with the image names. By default, we will use `train.txt` and `val.txt` in `VOC2012/Annotations/ImageSets/Main/`

## Configure the Training
To set up the training environment, follow these instructions:

1) Install Dependencies: If you haven't already, run `pip install -r requirements.txt`
2) Configure the system settings. In `src/configs/sys_config.py`, create a system config object such as the example below, with your desired settings.
```
object_detection_system_config = {
    "num_users": 3, 
    "dset": "pascal",
    "dump_dir": "./expt_dump/", # the path to place the results
    "dpath": "./datasets/pascal/VOCdevkit/VOC2012/", # the the location of the dataset
    # node_0 is a server currently
    # The device_ids dictionary depicts the GPUs on which the nodes reside.
    # For a single-GPU environment, the config will look as follows (as it follows a 0-based indexing):
    "device_ids": {"node_0": [0], "node_1": [0], "node_2": [0], "node_3": [0]},
    "samples_per_user": 100, 
    "train_label_distribution": "iid",
    "test_label_distribution": "iid",
    "folder_deletion_signal_path":"./expt_dump/folder_deletion.signal"
}
```
3) Configure the algorithm setting. In `src/configs/algo_config.py`, create an algo config object such as the example below, with your desired algorithm.
```
fedavg_object_detect = {
    "algo": "fedavg", # choose any algorithm we support
    "exp_id": "test",
    # Learning setup
    "epochs": 50,
    "model": "yolo",
    "model_lr": 1e-5,
    "batch_size": 8,
}
```
4) Initiate Training: `mpirun -n 4 python3 main.py`

> *Note: the `-n` flag should be followed by (number of desired users+ 1), for the server node*

> The training will proceed across the users as configured. Monitor printed or saved logs to track progress.

> Your result will be written into the `dump_dir` path specified in `sys_config.py`

## Additional Notes

* Ensure that the setup is correctly configured to avoid issues with client-server communication.
* If you encounter any issues or have suggestions, please open an issue on our [GitHub repository](https://github.com/aidecentralized/sonar).