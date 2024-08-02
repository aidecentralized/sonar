#!/bin/bash

# Define the args
host='scc-202.scc.bu.edu:50051' # This is the address of the server
model_arch='resnet18' # This is the model architecture
dataset='cifar10' # This is the dataset
algorithm='Fedavg' # This is the algorithm
num_clients=2 # This is the number of clients
num_trials=3 # This is the number of trials (number of seeds per hyperparameter set)
hparams_seed=1 # This is the seed for the hyperparameters
num_hparams=3 # This is the number of hyperparameter sets

# Run the sweep.py script with the defined variables
python sweep.py launch --host $host --model_arch $model_arch --dataset $dataset --algorithm $algorithm --num_clients $num_clients --num_trials $num_trials --hparams_seed $hparams_seed --num_hparams $num_hparams

# Aggregate the results and get the best hyperparameters set
python aggregate_results.py