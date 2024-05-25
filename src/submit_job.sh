#!/bin/bash
#SBATCH -o myScript.sh.log-%j
#SBATCH --gres=gpu:volta:2
#SBATCH -N 4
#SBATCH --ntasks-per-node=10

source /etc/profile

module load anaconda/2023a
module load mpi/openmpi-4.1.5
module load cuda/11.6

mpirun -np 40 --mca btl ^openib python main.py
