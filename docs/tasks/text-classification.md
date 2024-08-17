# Text Classification

## Overview
Our environment supports text classification tasks using Long Short-Term Memory (LSTM) networks. We provide an implementation of an LSTM-based text classification model using the AGNews dataset. The LSTM model is a type of recurrent neural network (RNN) that is particularly effective for sequential data such as text. In this project, we adapt an LSTM network to classify news articles into one of four categories: World, Sports, Business, and Sci/Tech. The implementation is designed to handle decentralized machine learning scenarios, where multiple users can train a shared model while keeping their data localized.

### Credit:
The AG's news topic classification dataset is constructed by Xiang Zhang (xiang.zhang@nyu.edu). It is used as a text classification benchmark in the following paper: Xiang Zhang, Junbo Zhao, Yann LeCun. Character-level Convolutional Networks for Text Classification. Advances in Neural Information Processing Systems 28 (NIPS 2015). This dataset is based on the [AGNews Dataset](http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html). 

## Dataset Preparation
We use the AGNews dataset, a popular benchmark in text classification tasks. Follow the steps below to download and prepare the dataset:
1) Download AGNews Data:

* Visit the [HuggingFace AGNews Dataset page](https://huggingface.co/datasets/fancyzhx/ag_news) to learn more about the dataset. 
* Download the dataset and extract it to your working directory. If you're using HuggingFace, it can be loaded in using the following code: `dataset = datasets.load_dataset('ag_news')`

2) Construct the Vocabulary

You must construct the vocabulary and preprocess the data by tokenizing the text and padding sequences to a uniform length. Here is the recommended script for preprocessing.
```
# Construct vocabulary from the dataset
words = Counter()

for example in dataset['train']['text']:
    processed_text = example.lower().translate(str.maketrans('', '', string.punctuation))
    for word in word_tokenize(processed_text):
        words[word] += 1

vocab = set(['<unk>', '<bos>', '<eos>', '<pad>'])
counter_threshold = 25

for char, cnt in words.items():
    if cnt > counter_threshold:
        vocab.add(char)

word2ind = {char: i for i, char in enumerate(vocab)}
ind2word = {i: char for char, i in word2ind.items()}
```

## Configure the Training
To set up the training environment, follow these instructions:
1) Install Dependencies: If you haven't already, run `pip install -r requirements.txt`.
2) Configure the system settings. In `src/configs/sys_config.py`, create a system config object such as the example below, with your desired settings.
```
text_classification_system_config = {
    "num_users": 3, 
    "dset": "agnews",
    "dump_dir": "./expt_dump/", # the path to place the results
    "dpath": "./datasets/agnews/", # the location of the dataset
    "device_ids": {"node_0": [0], "node_1": [0], "node_2": [0], "node_3": [0]},
    "samples_per_user": 1000, 
    "train_label_distribution": "iid",
    "test_label_distribution": "iid",
    "folder_deletion_signal_path":"./expt_dump/folder_deletion.signal"
}
```
3) Configure the algorithm settings. In `src/configs/algo_config.py`, create an algo config object such as the example below, with your desired algorithm.
```
fedavg_text_classify = {
    "algo": "fedavg", # choose any algorithm we support
    "exp_id": "test",
    # Learning setup
    "epochs": 10,
    "model": "lstm",
    "model_lr": 1e-3,
    "batch_size": 64,
}
```
4) Initiate Training: `mpirun -n 4 python3 main.py`
* *Note: the `-n` flag should be followed by (number of desired users + 1), for the server node.*
* The training will proceed across the users as configured. Monitor printed or saved logs to track progress.
* Your result will be written into the `dump_dir` path specified in `sys_config.py`.

## Additional Notes
* Ensure that the setup is correctly configured to avoid issues with client-server communication.
* If you encounter any issues or have suggestions, please open an issue on our [GitHub repository](https://github.com/aidecentralized/sonar).